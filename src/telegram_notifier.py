"""
Telegram notification system for arbitrage opportunities and trades
"""
from datetime import datetime
import requests
from data import secrets, settings

class TelegramNotifier:
    def __init__(self):
        self.bot_token = getattr(secrets, 'TELEGRAM_BOT_TOKEN', None)
        self.chat_id = getattr(secrets, 'TELEGRAM_CHAT_ID', None)
        self.enabled = self.bot_token and self.chat_id
    
    def send_message(self, message):
        """Send a message to Telegram"""
        if not self.enabled:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, json=payload, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending Telegram message: {e}")
            return False
    
    def notify_opportunity(self, exchange, asset, profit_pct, direction):
        """Notify about an arbitrage opportunity"""
        if not settings.PREMIUM_FEATURES_ENABLED:
            return
        
        message = (
            f"🚀 <b>Arbitrage Opportunity Found!</b>\n\n"
            f"Exchange: {exchange}\n"
            f"Asset: {asset}\n"
            f"Direction: {direction}\n"
            f"Profit: {profit_pct:.4f}%\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.send_message(message)
    
    def notify_trade_executed(self, exchange, asset, profit_eth, profit_usd, success):
        """Notify about trade execution"""
        if not settings.PREMIUM_FEATURES_ENABLED:
            return
        
        emoji = "✅" if success else "❌"
        message = (
            f"{emoji} <b>Trade Executed</b>\n\n"
            f"Exchange: {exchange}\n"
            f"Asset: {asset}\n"
            f"Profit: {profit_eth:.6f} ETH (${profit_usd:.2f})\n"
            f"Status: {'Success' if success else 'Failed'}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.send_message(message)
    
    def notify_daily_summary(self, stats):
        """Send daily profit summary"""
        if not settings.PREMIUM_FEATURES_ENABLED:
            return
        
        message = (
            f"📊 <b>Daily Trading Summary</b>\n\n"
            f"Total Profit: {stats['total_profit_eth']:.6f} ETH (${stats['total_profit_usd']:.2f})\n"
            f"Trades Executed: {stats['trades_executed']}\n"
            f"Successful Trades: {stats['successful_trades']}\n"
            f"Success Rate: {stats['success_rate']:.1f}%\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d')}"
        )
        self.send_message(message)
