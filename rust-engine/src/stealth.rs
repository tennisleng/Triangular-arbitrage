use rand::RngExt;

pub fn calculate_stealth_size(target_size: f64) -> f64 {
    let mut rng = rand::rng();
    // Add ±5% noise to make flow look retail, hiding from HFT competitors
    let noise = 1.0 + rng.random_range(-0.05..0.05);
    target_size * noise
}

pub fn get_competition_jitter_ms(win_rate: f64) -> u64 {
    let mut rng = rand::rng();
    // High win rate = low jitter (fast as possible)
    // Low win rate = high jitter (break the predictability)
    let max_jitter = if win_rate < 0.3 { 50 } else { 5 };
    rng.random_range(1..=max_jitter)
}

pub fn decay_opportunity(net_profit_bps: f64, age_ms: u64) -> f64 {
    if age_ms == 0 {
        return net_profit_bps;
    }
    let half_life_ms = 200.0;
    // Exponential decay
    net_profit_bps * (0.5_f64).powf(age_ms as f64 / half_life_ms)
}
