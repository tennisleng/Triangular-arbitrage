"""
API Models and Schemas
Request/response models for API endpoints
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import json


@dataclass
class APIResponse:
    """Standard API response wrapper"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        result = {
            'success': self.success,
            'timestamp': self.timestamp
        }
        if self.data is not None:
            result['data'] = self.data
        if self.error:
            result['error'] = self.error
        if self.message:
            result['message'] = self.message
        return result


@dataclass
class ProfitStats:
    """Profit statistics model"""
    total_profit_eth: float
    total_profit_usd: float
    trades_executed: int
    successful_trades: int
    success_rate: float
    last_updated: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass  
class TradeRecord:
    """Trade record model"""
    id: str
    timestamp: str
    asset: str
    direction: str  # 'forward' or 'backward'
    profit_eth: float
    profit_usd: float
    success: bool
    exchange: str = 'binance'
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SubscriptionInfo:
    """Subscription information model"""
    tier: str
    features: Dict[str, bool]
    expires_at: Optional[str]
    is_valid: bool
    days_remaining: Optional[int] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class APIKeyInfo:
    """API key information model"""
    key_prefix: str
    created_at: str
    last_used: Optional[str]
    requests_count: int
    active: bool
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ArbitrageOpportunity:
    """Live arbitrage opportunity model"""
    id: str
    asset: str
    direction: str  # 'forward' or 'backward'
    estimated_profit_pct: float
    estimated_profit_usd: float
    exchange: str
    timestamp: str
    expires_in_seconds: int = 5
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UsageStats:
    """API usage statistics model"""
    total_requests: int
    requests_today: int
    requests_this_month: int
    endpoints_used: Dict[str, int]
    average_response_time_ms: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HealthStatus:
    """Health check response model"""
    status: str  # 'healthy', 'degraded', 'unhealthy'
    version: str
    uptime_seconds: int
    subscription_tier: str
    exchange_connected: bool
    last_trade_at: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


# Request validation helpers
def validate_license_activation_request(data: dict) -> tuple:
    """
    Validate license activation request
    Returns: (is_valid: bool, error_message: Optional[str])
    """
    if not data:
        return False, 'Request body is required'
    
    license_key = data.get('license_key')
    if not license_key:
        return False, 'license_key is required'
    
    tier = data.get('tier', 'premium')
    valid_tiers = ['free', 'basic', 'premium', 'enterprise']
    if tier not in valid_tiers:
        return False, f'Invalid tier. Must be one of: {", ".join(valid_tiers)}'
    
    days = data.get('days', 30)
    if not isinstance(days, int) or days < 0 or days > 365:
        return False, 'days must be an integer between 0 and 365'
    
    return True, None


def validate_checkout_request(data: dict) -> tuple:
    """
    Validate checkout session request
    Returns: (is_valid: bool, error_message: Optional[str])
    """
    if not data:
        return False, 'Request body is required'
    
    tier = data.get('tier')
    valid_tiers = ['basic', 'premium', 'enterprise']
    if not tier or tier not in valid_tiers:
        return False, f'tier is required and must be one of: {", ".join(valid_tiers)}'
    
    success_url = data.get('success_url')
    if not success_url:
        return False, 'success_url is required'
    
    cancel_url = data.get('cancel_url')
    if not cancel_url:
        return False, 'cancel_url is required'
    
    return True, None


def serialize_response(data: Any) -> dict:
    """Convert response data to JSON-serializable format"""
    if hasattr(data, 'to_dict'):
        return data.to_dict()
    elif isinstance(data, list):
        return [serialize_response(item) for item in data]
    elif isinstance(data, dict):
        return {k: serialize_response(v) for k, v in data.items()}
    elif isinstance(data, datetime):
        return data.isoformat()
    return data
