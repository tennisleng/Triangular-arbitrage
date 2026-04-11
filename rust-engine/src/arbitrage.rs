use serde::Serialize;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::time::{sleep, Duration};

use crate::market::MarketData;
use crate::stealth;
use crate::DashboardEvent;

#[derive(Serialize, Clone, Debug)]
pub struct Opportunity {
    pub triangle: String,
    pub original_profit_bps: f64,
    pub decayed_profit_bps: f64,
}

pub struct ArbitrageEngine {
    market_data: Arc<MarketData>,
    tx_dashboard: broadcast::Sender<DashboardEvent>,
}

impl ArbitrageEngine {
    pub fn new(market_data: Arc<MarketData>, tx_dashboard: broadcast::Sender<DashboardEvent>) -> Self {
        Self {
            market_data,
            tx_dashboard,
        }
    }

    pub async fn run_loop(&self) {
        // High frequency loop evaluating pre-computed triangles
        let triangles = vec![
            ("ETH/USDT", "ETH/BTC", "BTC/USDT", 0.001), 
            ("SOL/USDT", "SOL/BTC", "BTC/USDT", 0.001), 
            ("LTC/USDT", "LTC/BTC", "BTC/USDT", 0.001), 
            ("SOL/USDT", "SOL/ETH", "ETH/USDT", 0.001), 
        ];

        let mut win_rate = 0.5;

        loop {
            let start = std::time::Instant::now();

            for (leg1, leg2, leg3, fee) in &triangles {
                // Fetch books from lock-free map extremely fast
                let book1 = self.market_data.get_book(leg1);
                let book2 = self.market_data.get_book(leg2);
                let book3 = self.market_data.get_book(leg3);

                if let (Some(b1), Some(b2), Some(b3)) = (book1, book2, book3) {
                    if b1.bids.is_empty() || b2.bids.is_empty() || b3.bids.is_empty() {
                        continue;
                    }

                    // For the sake of demonstration, we approximate the cross rate
                    // B1 bid -> B2 ask -> B3 bid (e.g. USDT -> ETH -> BTC -> USDT)
                    let p1 = b1.bids[0].price; // Buy crypto with stable
                    let p2 = b2.asks[0].price; // Sell for BTC
                    let p3 = b3.bids[0].price; // Sell BTC for stable

                    let gross_rate = (1.0 / p1) * p2 * p3; 
                    let net_rate = gross_rate * (1.0_f64 - fee).powi(3);

                    let profit_bps = (net_rate - 1.0) * 10000.0;

                    if profit_bps > 2.0 {
                        let age_ms = 5; // simulated latency 5ms
                        let decayed_profit_bps = stealth::decay_opportunity(profit_bps, age_ms);

                        let triangle_str = format!("{}->{}->{}", leg1, leg2, leg3);

                        if decayed_profit_bps > 1.5 {
                            // Win rate starts competing
                            let jitter = stealth::get_competition_jitter_ms(win_rate);
                            sleep(Duration::from_millis(jitter)).await;

                            let _ = self.tx_dashboard.send(DashboardEvent::OpportunityFound(Opportunity {
                                triangle: triangle_str.clone(),
                                original_profit_bps: profit_bps,
                                decayed_profit_bps,
                            }));

                            let stealth_size = stealth::calculate_stealth_size(1.0);

                            // Simulate successful execution
                            win_rate = (win_rate * 0.9) + (1.0 * 0.1); 

                            let executed = DashboardEvent::TradeExecuted {
                                triangle: triangle_str,
                                profit_bps: decayed_profit_bps,
                                size: stealth_size,
                                toxicity_score: 1.0, // Low toxicity
                                latency_us: start.elapsed().as_micros() as u64,
                            };
                            let _ = self.tx_dashboard.send(executed);
                        } else {
                            // Opportunity lost due to decay (frontrun)
                            win_rate = (win_rate * 0.9) + (0.0 * 0.1);
                        }
                    }
                }
            }

            // Rust can spin this in less than a microsecond,
            // we sleep lightly to avoid melting the CPU for this demo
            sleep(Duration::from_millis(10)).await;
        }
    }
}
