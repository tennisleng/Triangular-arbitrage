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
