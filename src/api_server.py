"""
REST API server for monetization - allows external access to bot features
"""
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from src.model import Model
from src.subscription import SubscriptionManager
import os

app = Flask(__name__)
api = Api(app)

# Global instances
model_instance = None
subscription_manager = SubscriptionManager()

class ProfitStatsAPI(Resource):
    """API endpoint for profit statistics"""
    def get(self):
        if not subscription_manager.has_feature('api_access'):
            return {'error': 'API access requires premium subscription'}, 403
        
        if model_instance is None:
            return {'error': 'Bot not initialized'}, 500
        
        stats = model_instance.get_profit_stats()
        return jsonify(stats)

class TradeHistoryAPI(Resource):
    """API endpoint for trade history"""
    def get(self):
        if not subscription_manager.has_feature('api_access'):
            return {'error': 'API access requires premium subscription'}, 403
        
        # Read from logs or trade history file
        try:
            with open('logs.txt', 'r') as f:
                lines = f.readlines()
                # Return last 100 lines
                return jsonify({'trades': lines[-100:]})
        except:
            return jsonify({'trades': []})

class SubscriptionAPI(Resource):
    """API endpoint for subscription management"""
    def get(self):
        return jsonify({
            'tier': subscription_manager.get_tier(),
            'features': subscription_manager.license_data['features'],
            'expires_at': subscription_manager.license_data['expires_at']
        })
    
    def post(self):
        """Activate license (in production, validate against payment system)"""
        data = request.get_json()
        license_key = data.get('license_key')
        tier = data.get('tier', 'premium')
        days = data.get('days', 30)
        
        # In production, validate license_key against payment/database
        if subscription_manager.activate_license(license_key, tier, days):
            return jsonify({'status': 'success', 'tier': tier})
        return {'error': 'Invalid license key'}, 400

class HealthAPI(Resource):
    """Health check endpoint"""
    def get(self):
        return jsonify({
            'status': 'running',
            'subscription_tier': subscription_manager.get_tier()
        })

# Register API routes
api.add_resource(ProfitStatsAPI, '/api/stats')
api.add_resource(TradeHistoryAPI, '/api/trades')
api.add_resource(SubscriptionAPI, '/api/subscription')
api.add_resource(HealthAPI, '/api/health')

def start_api_server(port=5000, model=None):
    """Start the API server"""
    global model_instance
    model_instance = model
    
    print(f"Starting API server on port {port}")
    print(f"API endpoints:")
    print(f"  GET /api/stats - Profit statistics")
    print(f"  GET /api/trades - Trade history")
    print(f"  GET /api/subscription - Subscription info")
    print(f"  POST /api/subscription - Activate license")
    print(f"  GET /api/health - Health check")
    
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    start_api_server()
