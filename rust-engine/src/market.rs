use parking_lot::RwLock;
use rand::RngExt;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::time::{sleep, Duration};

use crate::DashboardEvent;

#[derive(Clone, Debug)]
pub struct PriceLevel {
    pub price: f64,
    pub qty: f64,
}

#[derive(Clone, Debug)]
pub struct OrderBook {
    pub pair: String,
    pub bids: Vec<PriceLevel>, // Sorted desc
    pub asks: Vec<PriceLevel>, // Sorted asc
    pub last_update_ns: u64,
}

pub struct MarketData {
    // RwLock per book ensures we don't block access to other pairs 
    // when updating one pair. Pre-allocated hashmap.
    pub books: HashMap<String, Arc<RwLock<OrderBook>>>,
}

impl MarketData {
    pub fn new() -> Self {
        let pairs = vec![
            "ETH/USDT", "BTC/USDT", "ETH/BTC", 
            "SOL/USDT", "SOL/BTC", "SOL/ETH",
            "LTC/USDT", "LTC/BTC", "LTC/ETH"
        ];
        
        let mut books = HashMap::new();
        for pair in pairs {
            books.insert(
                pair.to_string(),
                Arc::new(RwLock::new(OrderBook {
                    pair: pair.to_string(),
                    bids: Vec::new(),
                    asks: Vec::new(),
                    last_update_ns: 0,
                })),
            );
        }

        Self { books }
    }

    pub fn get_book(&self, pair: &str) -> Option<OrderBook> {
        self.books.get(pair).map(|lock| lock.read().clone())
    }
}

// Ultra-fast simulated exchange feed filling the order books
pub fn start_exchange_feed(
    market_data: Arc<MarketData>,
    tx_dashboard: broadcast::Sender<DashboardEvent>,
) {
    let base_prices: HashMap<&str, f64> = vec![
        ("ETH/USDT", 2500.0),
        ("BTC/USDT", 60000.0),
        ("ETH/BTC", 2500.0 / 60000.0),
        ("SOL/USDT", 140.0),
        ("SOL/BTC", 140.0 / 60000.0),
        ("SOL/ETH", 140.0 / 2500.0),
        ("LTC/USDT", 80.0),
        ("LTC/BTC", 80.0 / 60000.0),
        ("LTC/ETH", 80.0 / 2500.0),
    ]
    .into_iter()
    .collect();

    for (pair, book_lock) in market_data.books.iter() {
        let book_lock = book_lock.clone();
        let pair = pair.clone();
        let tx = tx_dashboard.clone();
        let base_price = *base_prices.get(pair.as_str()).unwrap();

        tokio::spawn(async move {
            let mut rng = rand::rng();
            loop {
                // Simulate order book changes
                let noise = 1.0 + rng.random_range(-0.002..0.002);
                let mid = base_price * noise;
                
                // Jump-style manipulation noise (spikes)
                let spread_multiplier = if rng.random_bool(0.05) { 5.0 } else { 1.0 }; // Occasional wide spread
                let spread = mid * 0.0005 * spread_multiplier;

                let mut bids = Vec::new();
                let mut asks = Vec::new();

                let mut current_bid = mid - spread / 2.0;
                let mut current_ask = mid + spread / 2.0;

                // Simulate toxic book: occasionally create huge blocks at far levels
                let is_toxic = rng.random_bool(0.1);

                for i in 0..5 {
                    let qty = if is_toxic && i == 0 {
                        rng.random_range(50.0..100.0) // Gigantic vanishing bid/ask
                    } else {
                        rng.random_range(0.1..5.0)
                    };
                    bids.push(PriceLevel { price: current_bid, qty });
                    asks.push(PriceLevel { price: current_ask, qty });
                    
                    current_bid -= mid * 0.0001;
                    current_ask += mid * 0.0001;
                }

                // Update lock
                {
                    let mut b = book_lock.write();
                    b.bids = bids.clone();
                    b.asks = asks.clone();
                    b.last_update_ns = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos() as u64;
                }

                // Send to dashboard occasionally to prevent flooding
                if rng.random_bool(0.2) {
                    let _ = tx.send(DashboardEvent::OrderBookUpdate {
                        pair: pair.clone(),
                        bids: bids.into_iter().map(|l| (l.price, l.qty)).collect(),
                        asks: asks.into_iter().map(|l| (l.price, l.qty)).collect(),
                    });
                }
                
                // Send toxicity alert
                if is_toxic && rng.random_bool(0.3) {
                    let _ = tx.send(DashboardEvent::ToxicityAlert {
                        pair: pair.clone(),
                        reason: "Imbalanced/Layered depth detected".to_string(),
                        score: rng.random_range(70.0..99.0),
                    });
                }

                sleep(Duration::from_millis(rng.random_range(5..50))).await; // Microsecond loops
            }
        });
    }
}
