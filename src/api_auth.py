"""
API Authentication and Rate Limiting Module
Provides API key validation, rate limiting, and tier-based access control
"""
import functools
import hashlib
import secrets
import time
import json
import os
from datetime import datetime, timedelta
from flask import request, jsonify, g

# Rate limit storage (in production, use Redis)
_rate_limit_storage = {}
_api_keys_file = 'api_keys.json'


class RateLimiter:
    """Rate limiter with tier-based limits"""
    
    # Rate limits by tier: (requests_per_minute, requests_per_day)
    TIER_LIMITS = {
        'free': (10, 100),
        'basic': (60, 1000),
        'premium': (300, 10000),
        'enterprise': (10000, 1000000)  # Effectively unlimited
    }
    
    def __init__(self):
        self.storage = _rate_limit_storage
    
    def _get_window_key(self, api_key: str, window: str) -> str:
        """Generate storage key for rate limit window"""
        return f"{api_key}:{window}"
    
    def check_rate_limit(self, api_key: str, tier: str) -> tuple:
        """
        Check if request is within rate limits
        Returns: (allowed: bool, remaining: int, reset_time: int)
        """
        limits = self.TIER_LIMITS.get(tier, self.TIER_LIMITS['free'])
        per_minute, per_day = limits
        
        now = time.time()
        minute_window = int(now / 60)
        day_window = int(now / 86400)
        
        minute_key = self._get_window_key(api_key, f"min:{minute_window}")
        day_key = self._get_window_key(api_key, f"day:{day_window}")
        
        # Check minute limit
        minute_count = self.storage.get(minute_key, 0)
        if minute_count >= per_minute:
            reset_time = (minute_window + 1) * 60
            return False, 0, int(reset_time - now)
        
        # Check daily limit
        day_count = self.storage.get(day_key, 0)
        if day_count >= per_day:
            reset_time = (day_window + 1) * 86400
            return False, 0, int(reset_time - now)
        
        # Increment counters
        self.storage[minute_key] = minute_count + 1
        self.storage[day_key] = day_count + 1
        
        # Clean old entries periodically
        self._cleanup_old_entries(now)
        
        remaining = min(per_minute - minute_count - 1, per_day - day_count - 1)
        return True, remaining, 60 - int(now % 60)
    
    def _cleanup_old_entries(self, now: float):
        """Remove expired rate limit entries"""
        if len(self.storage) > 10000:  # Only cleanup when storage is large
            current_minute = int(now / 60)
            current_day = int(now / 86400)
            keys_to_delete = []
            
            for key in self.storage:
                if ':min:' in key:
                    window = int(key.split(':')[-1])
                    if window < current_minute - 1:
                        keys_to_delete.append(key)
                elif ':day:' in key:
                    window = int(key.split(':')[-1])
                    if window < current_day - 1:
                        keys_to_delete.append(key)
            
            for key in keys_to_delete:
                del self.storage[key]


class APIKeyManager:
    """Manages API keys for authentication"""
    
    def __init__(self):
        self.keys_file = _api_keys_file
        self.keys = self._load_keys()
    
    def _load_keys(self) -> dict:
        """Load API keys from file"""
        if os.path.exists(self.keys_file):
            try:
                with open(self.keys_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_keys(self):
        """Save API keys to file"""
        try:
            with open(self.keys_file, 'w') as f:
                json.dump(self.keys, f, indent=2)
        except Exception as e:
            print(f"Error saving API keys: {e}")
    
    def generate_api_key(self, user_id: str = None) -> str:
        """Generate a new API key"""
        key = f"arb_{secrets.token_hex(24)}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        self.keys[key_hash] = {
            'user_id': user_id or 'default',
            'created_at': datetime.now().isoformat(),
            'last_used': None,
            'requests_count': 0,
            'active': True
        }
        self._save_keys()
        return key
    
    def validate_api_key(self, api_key: str) -> dict:
        """
        Validate API key and return key info
        Returns None if invalid
        """
        if not api_key:
            return None
        
        # For demo/testing: accept any key starting with 'demo-' or 'test-'
        if api_key.startswith('demo-') or api_key.startswith('test-'):
            return {
                'user_id': 'demo_user',
                'tier': 'premium',
                'active': True
            }
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_info = self.keys.get(key_hash)
        
        if key_info and key_info.get('active', True):
            # Update usage stats
            key_info['last_used'] = datetime.now().isoformat()
            key_info['requests_count'] = key_info.get('requests_count', 0) + 1
            self._save_keys()
            return key_info
        
        return None
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        if key_hash in self.keys:
            self.keys[key_hash]['active'] = False
            self._save_keys()
            return True
        return False
    
    def list_keys(self, user_id: str = None) -> list:
        """List all API keys for a user (returns masked keys)"""
        keys = []
        for key_hash, info in self.keys.items():
            if user_id is None or info.get('user_id') == user_id:
                keys.append({
                    'key_prefix': f"arb_...{key_hash[-8:]}",
                    'created_at': info.get('created_at'),
                    'last_used': info.get('last_used'),
                    'requests_count': info.get('requests_count', 0),
                    'active': info.get('active', True)
                })
        return keys


# Global instances
rate_limiter = RateLimiter()
api_key_manager = APIKeyManager()


def require_api_key(required_tier: str = 'free'):
    """
    Decorator to require API key authentication
    Also enforces rate limiting based on tier
    
    Args:
        required_tier: Minimum tier required ('free', 'basic', 'premium', 'enterprise')
    """
    tier_hierarchy = ['free', 'basic', 'premium', 'enterprise']
    
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # Get API key from header
            api_key = request.headers.get('X-API-Key')
            
            if not api_key:
                return jsonify({
                    'error': 'API key required',
                    'message': 'Please provide an API key in the X-API-Key header'
                }), 401
            
            # Validate API key
            key_info = api_key_manager.validate_api_key(api_key)
            if not key_info:
                return jsonify({
                    'error': 'Invalid API key',
                    'message': 'The provided API key is invalid or has been revoked'
                }), 401
            
            # Get subscription tier
            try:
                from src.subscription import SubscriptionManager
                sub_manager = SubscriptionManager()
                tier = sub_manager.get_tier()
            except:
                tier = key_info.get('tier', 'free')
            
            # Check tier requirements
            if tier_hierarchy.index(tier) < tier_hierarchy.index(required_tier):
                return jsonify({
                    'error': 'Insufficient tier',
                    'message': f'This endpoint requires {required_tier} tier or higher',
                    'current_tier': tier,
                    'required_tier': required_tier
                }), 403
            
            # Check rate limit
            allowed, remaining, reset_time = rate_limiter.check_rate_limit(api_key, tier)
            
            if not allowed:
                response = jsonify({
                    'error': 'Rate limit exceeded',
                    'message': 'Too many requests. Please try again later.',
                    'retry_after': reset_time
                })
                response.headers['X-RateLimit-Remaining'] = '0'
                response.headers['X-RateLimit-Reset'] = str(reset_time)
                response.headers['Retry-After'] = str(reset_time)
                return response, 429
            
            # Store user info in g for use in endpoint
            g.api_key = api_key
            g.user_id = key_info.get('user_id', 'unknown')
            g.tier = tier
            
            # Call the actual function
            response = f(*args, **kwargs)
            
            # Add rate limit headers to response
            if hasattr(response, 'headers'):
                response.headers['X-RateLimit-Remaining'] = str(remaining)
                response.headers['X-RateLimit-Reset'] = str(reset_time)
            
            return response
        
        return decorated_function
    return decorator


def log_api_request(endpoint: str, method: str, status_code: int, response_time_ms: float):
    """Log API request for analytics"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'response_time_ms': response_time_ms,
            'user_id': getattr(g, 'user_id', 'anonymous'),
            'tier': getattr(g, 'tier', 'unknown')
        }
        
        # Append to log file
        with open('api_requests.log', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except:
        pass  # Don't fail on logging errors
