use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use parking_lot::RwLock;
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::broadcast;
use tower_http::cors::{Any, CorsLayer};

mod arbitrage;
mod market;
mod stealth;

use market::{MarketData, OrderBook};
use arbitrage::{ArbitrageEngine, Opportunity};

#[derive(Clone)]
struct AppState {
    market_data: Arc<MarketData>,
    tx_dashboard: broadcast::Sender<DashboardEvent>,
}

#[derive(Serialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum DashboardEvent {
    OrderBookUpdate {
        pair: String,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
    },
    OpportunityFound(Opportunity),
    TradeExecuted {
        triangle: String,
        profit_bps: f64,
        size: f64,
        toxicity_score: f64,
        latency_us: u64,
    },
    ToxicityAlert {
        pair: String,
        reason: String,
        score: f64,
    },
}

#[tokio::main]
async fn main() {
    env_logger::init();
    
    let market_data = Arc::new(MarketData::new());
    let (tx_dashboard, _) = broadcast::channel::<DashboardEvent>(1000);
    
    let state = AppState {
        market_data: market_data.clone(),
        tx_dashboard: tx_dashboard.clone(),
    };

    // Spin up mock exchange feed (in reality, connect to Binance/Jump websocket)
    market::start_exchange_feed(market_data.clone(), tx_dashboard.clone());

    // Start Arbitrage Engine
    let engine = ArbitrageEngine::new(market_data.clone(), tx_dashboard.clone());
    tokio::spawn(async move {
        engine.run_loop().await;
    });

    let cors = CorsLayer::new().allow_origin(Any);

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .layer(cors)
        .with_state(state);

    println!("Starting HFT Engine Dashboard API on 0.0.0.0:8000...");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: AppState) {
    let mut rx = state.tx_dashboard.subscribe();
    while let Ok(event) = rx.recv().await {
        if let Ok(json) = serde_json::to_string(&event) {
            if socket.send(Message::Text(axum::extract::ws::Utf8Bytes::from(json))).await.is_err() {
                break;
            }
        }
    }
}
