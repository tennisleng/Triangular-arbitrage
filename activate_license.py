#!/usr/bin/env python3
"""
Quick script to activate a demo license for testing monetization features
"""
from src.subscription import SubscriptionManager

def main():
    print("=" * 60)
    print("License Activation Tool")
    print("=" * 60)
    
    sm = SubscriptionManager()
    
    print("\nAvailable tiers:")
    print("  1. Free (basic features)")
    print("  2. Basic ($9.99/month) - Telegram notifications")
    print("  3. Premium ($29.99/month) - API access + analytics")
    print("  4. Enterprise ($99.99/month) - Multi-exchange + custom")
    
    choice = input("\nSelect tier (1-4) [default: 3]: ").strip() or "3"
    
    tier_map = {
        "1": "free",
        "2": "basic",
        "3": "premium",
        "4": "enterprise"
    }
    
    tier = tier_map.get(choice, "premium")
    days = input("License duration in days [default: 30]: ").strip() or "30"
    
    try:
        days = int(days)
    except:
        days = 30
    
    license_key = f"demo-{tier}-{days}days"
    
    if sm.activate_license(license_key, tier, days):
        print(f"\n✅ License activated successfully!")
        print(f"   Tier: {tier}")
        print(f"   Duration: {days} days")
        print(f"   Features enabled:")
        for feature, enabled in sm.license_data['features'].items():
            status = "✓" if enabled else "✗"
            print(f"     {status} {feature}")
    else:
        print("\n❌ Failed to activate license")

if __name__ == "__main__":
    main()
