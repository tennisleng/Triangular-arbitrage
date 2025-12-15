"""
Payment Handler Module
Stripe integration for subscription payments and webhooks
"""
import json
import os
from datetime import datetime, timedelta

# Stripe is optional - only import if available
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False
    stripe = None


class PaymentHandler:
    """Handles Stripe payment integration for subscriptions"""
    
    # Pricing configuration
    PRICING = {
        'basic': {
            'name': 'Basic',
            'price_usd': 9.99,
            'price_id': os.environ.get('STRIPE_PRICE_BASIC', 'price_basic'),
            'features': ['telegram_notifications']
        },
        'premium': {
            'name': 'Premium',
            'price_usd': 29.99,
            'price_id': os.environ.get('STRIPE_PRICE_PREMIUM', 'price_premium'),
            'features': ['telegram_notifications', 'api_access', 'advanced_analytics', 'custom_strategies']
        },
        'enterprise': {
            'name': 'Enterprise',
            'price_usd': 99.99,
            'price_id': os.environ.get('STRIPE_PRICE_ENTERPRISE', 'price_enterprise'),
            'features': ['telegram_notifications', 'api_access', 'advanced_analytics', 'multi_exchange', 'custom_strategies']
        }
    }
    
    def __init__(self):
        self.stripe_api_key = os.environ.get('STRIPE_SECRET_KEY')
        self.webhook_secret = os.environ.get('STRIPE_WEBHOOK_SECRET')
        
        if STRIPE_AVAILABLE and self.stripe_api_key:
            stripe.api_key = self.stripe_api_key
            self.enabled = True
        else:
            self.enabled = False
    
    def create_checkout_session(self, tier: str, success_url: str, cancel_url: str, customer_email: str = None) -> dict:
        """
        Create a Stripe checkout session for subscription
        
        Args:
            tier: Subscription tier ('basic', 'premium', 'enterprise')
            success_url: URL to redirect on successful payment
            cancel_url: URL to redirect on cancelled payment
            customer_email: Optional customer email
            
        Returns:
            dict with session_id and checkout_url
        """
        if not self.enabled:
            # Return demo session if Stripe not configured
            return {
                'status': 'demo',
                'message': 'Stripe not configured. Use demo license activation.',
                'demo_license': f'demo-{tier}-{datetime.now().strftime("%Y%m%d")}',
                'tier': tier
            }
        
        pricing = self.PRICING.get(tier)
        if not pricing:
            return {'error': f'Invalid tier: {tier}'}
        
        try:
            session_params = {
                'payment_method_types': ['card'],
                'line_items': [{
                    'price': pricing['price_id'],
                    'quantity': 1
                }],
                'mode': 'subscription',
                'success_url': success_url,
                'cancel_url': cancel_url,
                'metadata': {
                    'tier': tier
                }
            }
            
            if customer_email:
                session_params['customer_email'] = customer_email
            
            session = stripe.checkout.Session.create(**session_params)
            
            return {
                'session_id': session.id,
                'checkout_url': session.url,
                'tier': tier,
                'price_usd': pricing['price_usd']
            }
        except Exception as e:
            return {'error': str(e)}
    
    def verify_webhook_signature(self, payload: bytes, signature: str) -> dict:
        """
        Verify Stripe webhook signature
        
        Args:
            payload: Raw request body
            signature: Stripe-Signature header
            
        Returns:
            Parsed event if valid, None if invalid
        """
        if not self.enabled or not self.webhook_secret:
            # In demo mode, parse without verification
            try:
                return json.loads(payload)
            except:
                return None
        
        try:
            event = stripe.Webhook.construct_event(
                payload, signature, self.webhook_secret
            )
            return event
        except Exception as e:
            print(f"Webhook signature verification failed: {e}")
            return None
    
    def process_webhook_event(self, event: dict) -> dict:
        """
        Process Stripe webhook event
        
        Handles:
        - checkout.session.completed: Activate subscription
        - customer.subscription.updated: Update subscription
        - customer.subscription.deleted: Cancel subscription
        - invoice.payment_failed: Handle failed payment
        
        Returns:
            dict with processing result
        """
        event_type = event.get('type')
        data = event.get('data', {}).get('object', {})
        
        result = {'event_type': event_type, 'processed': False}
        
        try:
            from src.subscription import SubscriptionManager
            sub_manager = SubscriptionManager()
        except:
            return {'error': 'Failed to load subscription manager'}
        
        if event_type == 'checkout.session.completed':
            # New subscription created
            tier = data.get('metadata', {}).get('tier', 'premium')
            customer_email = data.get('customer_email')
            
            # Activate license for 30 days
            license_key = f"stripe-{data.get('id', 'unknown')}"
            sub_manager.activate_license(license_key, tier, days=30)
            
            result['processed'] = True
            result['action'] = 'subscription_activated'
            result['tier'] = tier
            result['customer_email'] = customer_email
            
        elif event_type == 'customer.subscription.updated':
            # Subscription updated (upgrade/downgrade)
            status = data.get('status')
            
            if status == 'active':
                # Get tier from price metadata
                items = data.get('items', {}).get('data', [])
                if items:
                    price_id = items[0].get('price', {}).get('id')
                    tier = self._get_tier_from_price(price_id)
                    if tier:
                        sub_manager.activate_license(f"stripe-update", tier, days=30)
                        result['tier'] = tier
            
            result['processed'] = True
            result['action'] = 'subscription_updated'
            result['status'] = status
            
        elif event_type == 'customer.subscription.deleted':
            # Subscription cancelled
            sub_manager.activate_license('cancelled', 'free', days=0)
            result['processed'] = True
            result['action'] = 'subscription_cancelled'
            
        elif event_type == 'invoice.payment_failed':
            # Payment failed
            result['processed'] = True
            result['action'] = 'payment_failed'
            result['message'] = 'Payment failed, subscription may be suspended'
        
        return result
    
    def _get_tier_from_price(self, price_id: str) -> str:
        """Get tier name from Stripe price ID"""
        for tier, config in self.PRICING.items():
            if config['price_id'] == price_id:
                return tier
        return None
    
    def get_pricing_info(self) -> dict:
        """Get pricing information for display"""
        return {
            'currency': 'USD',
            'billing_period': 'monthly',
            'tiers': {
                tier: {
                    'name': info['name'],
                    'price': info['price_usd'],
                    'features': info['features']
                }
                for tier, info in self.PRICING.items()
            }
        }
    
    def get_customer_portal_url(self, customer_id: str, return_url: str) -> dict:
        """
        Create Stripe customer portal session for subscription management
        
        Args:
            customer_id: Stripe customer ID
            return_url: URL to return to after portal session
            
        Returns:
            dict with portal_url
        """
        if not self.enabled:
            return {
                'status': 'demo',
                'message': 'Stripe not configured. Manage subscription via API.'
            }
        
        try:
            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url
            )
            return {'portal_url': session.url}
        except Exception as e:
            return {'error': str(e)}


# Global instance
payment_handler = PaymentHandler()
