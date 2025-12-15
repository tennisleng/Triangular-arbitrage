"""
Subscription and licensing system for monetization
"""
import json
import os
from datetime import datetime, timedelta
from data import settings

class SubscriptionManager:
    def __init__(self):
        self.license_file = 'license.json'
        self.load_license()
    
    def load_license(self):
        """Load license information"""
        self.license_data = {
            'tier': 'free',  # free, basic, premium, enterprise
            'expires_at': None,
            'features': {
                'telegram_notifications': False,
                'api_access': False,
                'advanced_analytics': False,
                'multi_exchange': False,
                'custom_strategies': False
            }
        }
        
        if os.path.exists(self.license_file):
            try:
                with open(self.license_file, 'r') as f:
                    loaded = json.load(f)
                    self.license_data.update(loaded)
            except:
                pass
    
    def save_license(self):
        """Save license information"""
        try:
            with open(self.license_file, 'w') as f:
                json.dump(self.license_data, f, indent=2)
        except Exception as e:
            print(f"Error saving license: {e}")
    
    def is_valid(self):
        """Check if license is valid"""
        if self.license_data['expires_at']:
            try:
                expires = datetime.fromisoformat(self.license_data['expires_at'])
                return datetime.now() < expires
            except:
                return False
        return True
    
    def has_feature(self, feature):
        """Check if user has access to a feature"""
        if not self.is_valid():
            return False
        return self.license_data['features'].get(feature, False)
    
    def get_tier(self):
        """Get current subscription tier"""
        if not self.is_valid():
            return 'free'
        return self.license_data['tier']
    
    def activate_license(self, license_key, tier='premium', days=30):
        """Activate a license (for demo/testing)"""
        # In production, validate license_key against a server
        self.license_data['tier'] = tier
        self.license_data['expires_at'] = (datetime.now() + timedelta(days=days)).isoformat()
        
        # Set features based on tier
        if tier == 'free':
            self.license_data['features'] = {
                'telegram_notifications': False,
                'api_access': False,
                'advanced_analytics': False,
                'multi_exchange': False,
                'custom_strategies': False
            }
        elif tier == 'basic':
            self.license_data['features'] = {
                'telegram_notifications': True,
                'api_access': False,
                'advanced_analytics': False,
                'multi_exchange': False,
                'custom_strategies': False
            }
        elif tier == 'premium':
            self.license_data['features'] = {
                'telegram_notifications': True,
                'api_access': True,
                'advanced_analytics': True,
                'multi_exchange': False,
                'custom_strategies': True
            }
        elif tier == 'enterprise':
            self.license_data['features'] = {
                'telegram_notifications': True,
                'api_access': True,
                'advanced_analytics': True,
                'multi_exchange': True,
                'custom_strategies': True
            }
        
        self.save_license()
        return True
    
    def check_premium_features(self):
        """Update settings based on license"""
        if self.has_feature('telegram_notifications'):
            settings.PREMIUM_FEATURES_ENABLED = True
        else:
            settings.PREMIUM_FEATURES_ENABLED = False
    
    def get_days_remaining(self):
        """Get number of days remaining on subscription"""
        if not self.license_data['expires_at']:
            return None
        try:
            expires = datetime.fromisoformat(self.license_data['expires_at'])
            remaining = (expires - datetime.now()).days
            return max(0, remaining)
        except:
            return None
    
    def upgrade_tier(self, new_tier: str, days: int = 30):
        """Upgrade to a new tier"""
        valid_tiers = ['free', 'basic', 'premium', 'enterprise']
        if new_tier not in valid_tiers:
            return False
        
        current_tier_idx = valid_tiers.index(self.license_data['tier'])
        new_tier_idx = valid_tiers.index(new_tier)
        
        if new_tier_idx <= current_tier_idx:
            return False  # Can only upgrade, not downgrade
        
        return self.activate_license(f'upgrade-{new_tier}', new_tier, days)
    
    def get_subscription_details(self):
        """Get detailed subscription information"""
        return {
            'tier': self.get_tier(),
            'features': self.license_data['features'],
            'expires_at': self.license_data['expires_at'],
            'is_valid': self.is_valid(),
            'days_remaining': self.get_days_remaining()
        }
