# Triangular Arbitrage Engine

An institutional-grade, ultra-low latency triangular arbitrage engine. Previously scaled in Python for prototyping, the core execution path is now rewritten completely in high-performance **Rust**, equipped with a real-time **React/Vite** web dashboard.

## Overview
This platform targets Malthusian HFT competition, specifically structured to eat into the margins of tier-1 quantitative prop firms (e.g. Jump Trading, Wintermute). By embracing predictability randomization and structural lock-free concurrency, we beat the tape while appearing as organic flow.

## Architecture

The system is now segregated into three layers:

```
Triangular-arbitrage/
├── rust-engine/          # (Active) Ultra-fast core trading engine
│   ├── src/main.rs       # Async Tokio runtime, axum WS endpoints
│   ├── src/market.rs     # Lock-free DashMap/RwLock Order Books
│   ├── src/arbitrage.rs  # Graph routing, spread calcs (fixed types)
│   └── src/stealth.rs    # Anti-frontrunning / execution fuzzing
├── dashboard-web/        # (Active) Trader's Terminal UI
│   ├── src/App.tsx       # Live ticks, PnL heatmaps, Threat detection 
│   └── ...               # React + Tailwind + Vite + Recharts
└── python-legacy/        # (Deprecated) Original mathematical prototype
```

### 1. Ultra-Low Latency Rust Engine
The execution path runs purely in memory without garbage collector stutter:
*   **Zero-Copy Routing:** Heavy decimal crates bypassed during intermediate triage. F64 matrix math runs via vectorized loops.
*   **Lock-Free State:** `RwLock`-wrapped Atomically Reference Counted pointers ensure deep book ingestion never blocks the arbitrage detection loop.
*   **WebSockets:** An embedded `axum` TCP listener streams trade and toxicity events via broadcast channels directly to the frontend.

### 2. Competitive Stealth & Toxicity Filtering
Raw speed is secondary to obfuscation in highly saturated HFT pools:
*   **Dynamic Jittering:** The engine actively calculates a trailing *Win Rate*. If competition intensifies, we introduce randomized microsecond jitters. This breaks the time-correlation signature.
*   **Size Obfuscation:** The stealth model applies a continuous ±5% noise parameter (`rand`) against execution size. Retail flow is messy; identical size flow is algorithmic. We camouflage.
*   **Toxic Order Book Threat Detection:** Discovers spoofing (fake layered liquidity), disappearing depth, and skewed imbalance ratios in real time.

### 3. Trader's Dashboard
Built over React + TailwindCSS + Vite logic. Connects locally over WS.
*   Live Order Book Heatmaps (fast-scroll depths).
*   Live Execution Alpha Chart (scrolling area chart of real basis points captured).
*   Live Threat Detection log. 

## Quick Start

### Backend (Rust)
```bash
cd rust-engine
cargo run --release
```
Will start the engine on `0.0.0.0:8000`.

### Frontend (Dashboard)
```bash
cd dashboard-web
npm install
npm run dev
```
Will launch the UI.

## Testing & CI
Continuous Integration handles rigorous build checks for the new stack while automatically validating memory safety.

```bash
# Rust
cd rust-engine && cargo test

# Web
cd dashboard-web && npm run build
```

## Lineage & License
Built from zero for competitive execution logic. 
[GPL-3.0](LICENSE)
