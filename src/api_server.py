"""
REST API Server for Monetization
Comprehensive API with authentication, rate limiting, and full monetization features
"""
import os
import time
from datetime import datetime
from flask import Flask, jsonify, request, g
from flask_restful import Api, Resource
from flask_cors import CORS

# Swagger documentation
try:
    from flasgger import Swagger, swag_from
    SWAGGER_AVAILABLE = True
except ImportError:
    SWAGGER_AVAILABLE = False
    def swag_from(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

from src.api_auth import require_api_key, api_key_manager, rate_limiter, log_api_request
from src.api_models import (
    APIResponse, ProfitStats, SubscriptionInfo, HealthStatus,
    validate_license_activation_request, validate_checkout_request,
    serialize_response
)
from src.payment_handler import payment_handler
from src.subscription import SubscriptionManager

# App configuration
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['SWAGGER'] = {
    'title': 'Triangular Arbitrage Bot API',
    'description': 'REST API for cryptocurrency triangular arbitrage bot monetization',
    'version': '1.0.0',
    'termsOfService': '',
    'contact': {
        'name': 'API Support',
        'email': 'support@example.com'
    }
}

# Enable CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize Swagger if available
if SWAGGER_AVAILABLE:
    swagger = Swagger(app)

api = Api(app)

# Global instances
model_instance = None
subscription_manager = SubscriptionManager()
start_time = time.time()
API_VERSION = '1.0.0'


# ============================================================================
# Health & Status Endpoints
# ============================================================================

class HealthAPI(Resource):
    """Health check endpoint - no authentication required"""
    
    def get(self):
        """
        Health check endpoint
        ---
        tags:
          - Status
        responses:
          200:
            description: API is healthy
        """
        uptime = int(time.time() - start_time)
        
        # Check exchange connection
        exchange_connected = False
        last_trade = None
        if model_instance:
            try:
                exchange_connected = model_instance.binance is not None
                if hasattr(model_instance, 'last_trade_time'):
                    last_trade = model_instance.last_trade_time
            except:
                pass
        
        status = HealthStatus(
            status='healthy',
            version=API_VERSION,
            uptime_seconds=uptime,
            subscription_tier=subscription_manager.get_tier(),
            exchange_connected=exchange_connected,
            last_trade_at=last_trade
        )
        
        return jsonify(status.to_dict())


class VersionAPI(Resource):
    """API version information"""
    
    def get(self):
        """
        Get API version and info
        ---
        tags:
          - Status
        responses:
          200:
            description: API version information
        """
        return jsonify({
            'name': 'Triangular Arbitrage Bot API',
            'version': API_VERSION,
            'docs': '/api/docs' if SWAGGER_AVAILABLE else None,
            'endpoints': {
                'health': '/api/v1/health',
                'stats': '/api/v1/stats',
                'trades': '/api/v1/trades',
                'subscription': '/api/v1/subscription',
                'pricing': '/api/v1/pricing'
            }
        })


# ============================================================================
# Stats & Analytics Endpoints
# ============================================================================

class ProfitStatsAPI(Resource):
    """Profit statistics endpoint"""
    
    @require_api_key('basic')
    def get(self):
        """
        Get profit statistics
        ---
        tags:
          - Statistics
        security:
          - ApiKeyAuth: []
        responses:
          200:
            description: Profit statistics
          401:
            description: Unauthorized
          403:
            description: Insufficient subscription tier
        """
        if model_instance is None:
            # Return demo stats if bot not running
            stats = {
                'total_profit_eth': 0.0,
                'total_profit_usd': 0.0,
                'trades_executed': 0,
                'successful_trades': 0,
                'success_rate': 0.0,
                'demo_mode': True
            }
        else:
            stats = model_instance.get_profit_stats()
        
        response = APIResponse(success=True, data=stats)
        return jsonify(response.to_dict())


class DetailedStatsAPI(Resource):
    """Detailed analytics endpoint - Premium feature"""
    
    @require_api_key('premium')
    def get(self):
        """
        Get detailed analytics (Premium)
        ---
        tags:
          - Statistics
        security:
          - ApiKeyAuth: []
        responses:
          200:
            description: Detailed analytics
          403:
            description: Premium subscription required
        """
        basic_stats = {}
        if model_instance:
            basic_stats = model_instance.get_profit_stats()
        
        # Load additional analytics
        analytics = {
            **basic_stats,
            'hourly_breakdown': self._get_hourly_breakdown(),
            'top_performing_pairs': self._get_top_pairs(),
            'average_profit_per_trade': self._calculate_avg_profit(basic_stats),
            'profit_trend': self._get_profit_trend(),
            'tier': g.tier
        }
        
        response = APIResponse(success=True, data=analytics)
        return jsonify(response.to_dict())
    
    def _get_hourly_breakdown(self):
        """Get hourly trade breakdown"""
        # In production, load from database
        return {'note': 'Hourly data available after trades execute'}
    
    def _get_top_pairs(self):
        """Get top performing trading pairs"""
        return {'note': 'Pair data available after trades execute'}
    
    def _calculate_avg_profit(self, stats):
        """Calculate average profit per trade"""
        if stats.get('trades_executed', 0) > 0:
            return stats.get('total_profit_usd', 0) / stats['trades_executed']
        return 0.0
    
    def _get_profit_trend(self):
        """Get profit trend over time"""
        return {'note': 'Trend data available after multiple trades'}


# ============================================================================
# Trade Endpoints  
# ============================================================================

class TradeHistoryAPI(Resource):
    """Trade history endpoint"""
    
    @require_api_key('basic')
    def get(self):
        """
        Get trade history
        ---
        tags:
          - Trades
        parameters:
          - name: limit
            in: query
            type: integer
            default: 100
            description: Number of trades to return
          - name: offset
            in: query
            type: integer
            default: 0
            description: Offset for pagination
        security:
          - ApiKeyAuth: []
        responses:
          200:
            description: Trade history
        """
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        # Cap limit at 500
        limit = min(limit, 500)
        
        trades = []
        try:
            with open('logs.txt', 'r') as f:
                lines = f.readlines()
                # Filter for trade-related lines
                trade_lines = [l.strip() for l in lines if 'Arbitrage' in l or 'profit' in l.lower()]
                trades = trade_lines[offset:offset + limit]
        except:
            pass
        
        response = APIResponse(
            success=True,
            data={
                'trades': trades,
                'count': len(trades),
                'limit': limit,
                'offset': offset
            }
        )
        return jsonify(response.to_dict())


class LiveOpportunitiesAPI(Resource):
    """Live arbitrage opportunities - Enterprise feature"""
    
    @require_api_key('enterprise')
    def get(self):
        """
        Get live arbitrage opportunities (Enterprise)
        ---
        tags:
          - Trades
        security:
          - ApiKeyAuth: []
        responses:
          200:
            description: Live opportunities
          403:
            description: Enterprise subscription required
        """
        opportunities = []
        
        if model_instance:
            # Scan for current opportunities
            try:
                from data import tokens
                for token in tokens.SYMBOLS[:10]:  # Limit to first 10 for performance
                    forward_profit = model_instance.estimate_arbitrage_forward(
                        model_instance.binance, token
                    )
                    if forward_profit > 0:
                        eth_price = model_instance.get_price(
                            model_instance.binance, 'ETH', 'USDT', mode='average'
                        ) or 2000
                        opportunities.append({
                            'asset': token,
                            'direction': 'forward',
                            'estimated_profit_pct': forward_profit,
                            'estimated_profit_usd': forward_profit * eth_price / 100,
                            'exchange': 'binance',
                            'timestamp': datetime.now().isoformat()
                        })
            except Exception as e:
                pass
        
        response = APIResponse(
            success=True,
            data={
                'opportunities': opportunities,
                'count': len(opportunities),
                'scanned_at': datetime.now().isoformat()
            }
        )
        return jsonify(response.to_dict())


# ============================================================================
# Subscription & Licensing Endpoints
# ============================================================================

class SubscriptionAPI(Resource):
    """Subscription management endpoint"""
    
    def get(self):
        """
        Get subscription information
        ---
        tags:
          - Subscription
        responses:
          200:
            description: Subscription information
        """
        sub_info = {
            'tier': subscription_manager.get_tier(),
            'features': subscription_manager.license_data['features'],
            'expires_at': subscription_manager.license_data['expires_at'],
            'is_valid': subscription_manager.is_valid()
        }
        
        # Calculate days remaining
        if sub_info['expires_at']:
            try:
                expires = datetime.fromisoformat(sub_info['expires_at'])
                days_remaining = (expires - datetime.now()).days
                sub_info['days_remaining'] = max(0, days_remaining)
            except:
                pass
        
        response = APIResponse(success=True, data=sub_info)
        return jsonify(response.to_dict())


class ActivateLicenseAPI(Resource):
    """License activation endpoint"""
    
    def post(self):
        """
        Activate a license
        ---
        tags:
          - Subscription
        parameters:
          - in: body
            name: body
            schema:
              type: object
              required:
                - license_key
              properties:
                license_key:
                  type: string
                tier:
                  type: string
                  enum: [free, basic, premium, enterprise]
                  default: premium
                days:
                  type: integer
                  default: 30
        responses:
          200:
            description: License activated
          400:
            description: Invalid request
        """
        data = request.get_json() or {}
        
        # Validate request
        is_valid, error = validate_license_activation_request(data)
        if not is_valid:
            response = APIResponse(success=False, error=error)
            return jsonify(response.to_dict()), 400
        
        license_key = data.get('license_key')
        tier = data.get('tier', 'premium')
        days = data.get('days', 30)
        
        # Activate license
        if subscription_manager.activate_license(license_key, tier, days):
            response = APIResponse(
                success=True,
                message=f'License activated successfully for {days} days',
                data={
                    'tier': tier,
                    'expires_at': subscription_manager.license_data['expires_at'],
                    'features': subscription_manager.license_data['features']
                }
            )
            return jsonify(response.to_dict())
        
        response = APIResponse(success=False, error='Failed to activate license')
        return jsonify(response.to_dict()), 400


class PricingAPI(Resource):
    """Pricing information endpoint"""
    
    def get(self):
        """
        Get pricing information
        ---
        tags:
          - Subscription
        responses:
          200:
            description: Pricing tiers and features
        """
        pricing = payment_handler.get_pricing_info()
        
        # Add feature descriptions
        feature_descriptions = {
            'telegram_notifications': 'Real-time Telegram alerts for trades and opportunities',
            'api_access': 'Full REST API access with rate limiting',
            'advanced_analytics': 'Detailed profit analytics and reporting',
            'multi_exchange': 'Support for multiple cryptocurrency exchanges',
            'custom_strategies': 'Custom trading strategy configuration'
        }
        
        pricing['feature_descriptions'] = feature_descriptions
        
        response = APIResponse(success=True, data=pricing)
        return jsonify(response.to_dict())


# ============================================================================
# Payment Endpoints
# ============================================================================

class CheckoutAPI(Resource):
    """Payment checkout endpoint"""
    
    def post(self):
        """
        Create checkout session
        ---
        tags:
          - Payment
        parameters:
          - in: body
            name: body
            schema:
              type: object
              required:
                - tier
                - success_url
                - cancel_url
              properties:
                tier:
                  type: string
                  enum: [basic, premium, enterprise]
                success_url:
                  type: string
                cancel_url:
                  type: string
                email:
                  type: string
        responses:
          200:
            description: Checkout session created
          400:
            description: Invalid request
        """
        data = request.get_json() or {}
        
        # Validate request
        is_valid, error = validate_checkout_request(data)
        if not is_valid:
            response = APIResponse(success=False, error=error)
            return jsonify(response.to_dict()), 400
        
        result = payment_handler.create_checkout_session(
            tier=data['tier'],
            success_url=data['success_url'],
            cancel_url=data['cancel_url'],
            customer_email=data.get('email')
        )
        
        if 'error' in result:
            response = APIResponse(success=False, error=result['error'])
            return jsonify(response.to_dict()), 400
        
        response = APIResponse(success=True, data=result)
        return jsonify(response.to_dict())


class PaymentWebhookAPI(Resource):
    """Stripe webhook endpoint"""
    
    def post(self):
        """
        Handle Stripe webhook events
        ---
        tags:
          - Payment
        responses:
          200:
            description: Webhook processed
          400:
            description: Invalid webhook
        """
        payload = request.get_data()
        signature = request.headers.get('Stripe-Signature', '')
        
        # Verify and parse event
        event = payment_handler.verify_webhook_signature(payload, signature)
        if not event:
            return jsonify({'error': 'Invalid webhook signature'}), 400
        
        # Process event
        result = payment_handler.process_webhook_event(event)
        
        return jsonify({
            'received': True,
            'processed': result.get('processed', False),
            'action': result.get('action')
        })


# ============================================================================
# API Key Management Endpoints
# ============================================================================

class APIKeysAPI(Resource):
    """API key management endpoint"""
    
    @require_api_key('basic')
    def get(self):
        """
        List API keys
        ---
        tags:
          - API Keys
        security:
          - ApiKeyAuth: []
        responses:
          200:
            description: List of API keys
        """
        keys = api_key_manager.list_keys(user_id=g.user_id)
        response = APIResponse(success=True, data={'keys': keys})
        return jsonify(response.to_dict())
    
    @require_api_key('basic')
    def post(self):
        """
        Generate new API key
        ---
        tags:
          - API Keys
        security:
          - ApiKeyAuth: []
        responses:
          201:
            description: API key created
        """
        new_key = api_key_manager.generate_api_key(user_id=g.user_id)
        
        response = APIResponse(
            success=True,
            message='API key generated. Store it securely - it cannot be retrieved later.',
            data={'api_key': new_key}
        )
        return jsonify(response.to_dict()), 201
    
    @require_api_key('basic')
    def delete(self):
        """
        Revoke an API key
        ---
        tags:
          - API Keys
        security:
          - ApiKeyAuth: []
        parameters:
          - in: body
            name: body
            schema:
              type: object
              required:
                - api_key
              properties:
                api_key:
                  type: string
        responses:
          200:
            description: API key revoked
        """
        data = request.get_json() or {}
        key_to_revoke = data.get('api_key')
        
        if not key_to_revoke:
            response = APIResponse(success=False, error='api_key is required')
            return jsonify(response.to_dict()), 400
        
        if api_key_manager.revoke_api_key(key_to_revoke):
            response = APIResponse(success=True, message='API key revoked')
            return jsonify(response.to_dict())
        
        response = APIResponse(success=False, error='API key not found')
        return jsonify(response.to_dict()), 404


# ============================================================================
# Usage Analytics Endpoints
# ============================================================================

class UsageAnalyticsAPI(Resource):
    """API usage analytics - Premium feature"""
    
    @require_api_key('premium')
    def get(self):
        """
        Get API usage statistics (Premium)
        ---
        tags:
          - Analytics
        security:
          - ApiKeyAuth: []
        responses:
          200:
            description: Usage statistics
        """
        # Load usage stats from log file
        usage_data = self._load_usage_data()
        
        response = APIResponse(success=True, data=usage_data)
        return jsonify(response.to_dict())
    
    def _load_usage_data(self):
        """Load usage data from log file"""
        total_requests = 0
        requests_today = 0
        endpoints_used = {}
        response_times = []
        
        today = datetime.now().date().isoformat()
        
        try:
            with open('api_requests.log', 'r') as f:
                for line in f:
                    try:
                        import json
                        entry = json.loads(line.strip())
                        total_requests += 1
                        
                        if entry.get('timestamp', '').startswith(today):
                            requests_today += 1
                        
                        endpoint = entry.get('endpoint', 'unknown')
                        endpoints_used[endpoint] = endpoints_used.get(endpoint, 0) + 1
                        
                        if 'response_time_ms' in entry:
                            response_times.append(entry['response_time_ms'])
                    except:
                        pass
        except:
            pass
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            'total_requests': total_requests,
            'requests_today': requests_today,
            'endpoints_used': endpoints_used,
            'average_response_time_ms': round(avg_response_time, 2)
        }


# ============================================================================
# Legacy API Routes (backward compatibility)
# ============================================================================

class LegacyStatsAPI(Resource):
    """Legacy stats endpoint for backward compatibility"""
    
    def get(self):
        if not subscription_manager.has_feature('api_access'):
            return {'error': 'API access requires premium subscription'}, 403
        
        if model_instance is None:
            return {'error': 'Bot not initialized'}, 500
        
        stats = model_instance.get_profit_stats()
        return jsonify(stats)


class LegacyTradesAPI(Resource):
    """Legacy trades endpoint for backward compatibility"""
    
    def get(self):
        if not subscription_manager.has_feature('api_access'):
            return {'error': 'API access requires premium subscription'}, 403
        
        try:
            with open('logs.txt', 'r') as f:
                lines = f.readlines()
                return jsonify({'trades': lines[-100:]})
        except:
            return jsonify({'trades': []})


class LegacySubscriptionAPI(Resource):
    """Legacy subscription endpoint for backward compatibility"""
    
    def get(self):
        return jsonify({
            'tier': subscription_manager.get_tier(),
            'features': subscription_manager.license_data['features'],
            'expires_at': subscription_manager.license_data['expires_at']
        })
    
    def post(self):
        data = request.get_json()
        license_key = data.get('license_key')
        tier = data.get('tier', 'premium')
        days = data.get('days', 30)
        
        if subscription_manager.activate_license(license_key, tier, days):
            return jsonify({'status': 'success', 'tier': tier})
        return {'error': 'Invalid license key'}, 400


class LegacyHealthAPI(Resource):
    """Legacy health endpoint for backward compatibility"""
    
    def get(self):
        return jsonify({
            'status': 'running',
            'subscription_tier': subscription_manager.get_tier()
        })


# ============================================================================
# Register Routes
# ============================================================================

# API v1 Routes (new)
api.add_resource(HealthAPI, '/api/v1/health')
api.add_resource(VersionAPI, '/api/v1/version', '/api/v1')
api.add_resource(ProfitStatsAPI, '/api/v1/stats')
api.add_resource(DetailedStatsAPI, '/api/v1/stats/detailed')
api.add_resource(TradeHistoryAPI, '/api/v1/trades')
api.add_resource(LiveOpportunitiesAPI, '/api/v1/opportunities')
api.add_resource(SubscriptionAPI, '/api/v1/subscription')
api.add_resource(ActivateLicenseAPI, '/api/v1/subscription/activate')
api.add_resource(PricingAPI, '/api/v1/pricing')
api.add_resource(CheckoutAPI, '/api/v1/payment/checkout')
api.add_resource(PaymentWebhookAPI, '/api/v1/payment/webhook')
api.add_resource(APIKeysAPI, '/api/v1/user/api-keys')
api.add_resource(UsageAnalyticsAPI, '/api/v1/analytics/usage')

# Legacy Routes (backward compatibility)
api.add_resource(LegacyStatsAPI, '/api/stats')
api.add_resource(LegacyTradesAPI, '/api/trades')
api.add_resource(LegacySubscriptionAPI, '/api/subscription')
api.add_resource(LegacyHealthAPI, '/api/health')


# ============================================================================
# Request Hooks
# ============================================================================

@app.before_request
def before_request():
    """Log request start time"""
    g.start_time = time.time()


@app.after_request
def after_request(response):
    """Log request and add headers"""
    # Calculate response time
    if hasattr(g, 'start_time'):
        response_time = (time.time() - g.start_time) * 1000
        
        # Log API request
        if request.path.startswith('/api/'):
            log_api_request(
                endpoint=request.path,
                method=request.method,
                status_code=response.status_code,
                response_time_ms=response_time
            )
    
    # Add CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-API-Key'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    
    return response


# ============================================================================
# Server Startup
# ============================================================================

def start_api_server(port=5000, model=None, debug=False):
    """Start the API server"""
    global model_instance
    model_instance = model
    
    print(f"\n{'='*60}")
    print(f"  Triangular Arbitrage Bot API v{API_VERSION}")
    print(f"{'='*60}")
    print(f"\n  Server starting on port {port}")
    print(f"  Subscription tier: {subscription_manager.get_tier()}")
    
    print(f"\n  API v1 Endpoints:")
    print(f"  {'─'*50}")
    print(f"  GET  /api/v1/health          - Health check")
    print(f"  GET  /api/v1/version         - Version information")
    print(f"  GET  /api/v1/stats           - Profit statistics")
    print(f"  GET  /api/v1/stats/detailed  - Detailed analytics (Premium)")
    print(f"  GET  /api/v1/trades          - Trade history")
    print(f"  GET  /api/v1/opportunities   - Live opportunities (Enterprise)")
    print(f"  GET  /api/v1/subscription    - Subscription info")
    print(f"  POST /api/v1/subscription/activate - Activate license")
    print(f"  GET  /api/v1/pricing         - Pricing information")
    print(f"  POST /api/v1/payment/checkout - Create checkout")
    print(f"  POST /api/v1/payment/webhook - Stripe webhook")
    print(f"  GET  /api/v1/user/api-keys   - List API keys")
    print(f"  POST /api/v1/user/api-keys   - Generate API key")
    print(f"  GET  /api/v1/analytics/usage - Usage stats (Premium)")
    
    if SWAGGER_AVAILABLE:
        print(f"\n  Documentation: http://localhost:{port}/apidocs")
    
    print(f"\n{'='*60}\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)


if __name__ == '__main__':
    start_api_server()
