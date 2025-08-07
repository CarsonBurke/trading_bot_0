use rand::{rng, seq::IndexedRandom, Rng};
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};

use crate::{
    constants::TICKERS, custom_rl::{
        ActorCritic, History, PPOBuffer, TradingEnvironment, ENTROPY_COEF, EPSILON, LEARNING_RATE,
        VALUE_COEF,
    }, data::historical::get_historical_data
};

pub fn train() {
    let mapped_historical = get_historical_data();

    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let model = ActorCritic::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, LEARNING_RATE).unwrap();

    let mut env = TradingEnvironment::new(mapped_historical.len());
    let mut buffer = PPOBuffer::new();

    for generation in 0..1000 {
        let mut history = History::new(mapped_historical.len());

        for (ticker_index, bars) in mapped_historical.iter().enumerate() {
            let prices = bars.iter().map(|bar| bar.close).collect::<Vec<f64>>();

            let mut obs = env.reset(ticker_index, prices.clone());
            let mut episode_reward = 0.0;

            // Collect trajectory
            for _ in env.current_step..prices.len() - 1 {
                let (action_logits, value) = tch::no_grad(|| model.forward(&obs));

                let probs = action_logits.softmax(-1, tch::Kind::Float);
                let action = probs.multinomial(1, true).int64_value(&[0]);
                let log_prob =
                    probs
                        .log()
                        .gather(1, &Tensor::from_slice(&[action]).view([1, 1]), false);

                buffer.add(
                    obs.shallow_clone(),
                    action,
                    0.0, // Will be updated with actual reward
                    value.double_value(&[]),
                    log_prob.double_value(&[]),
                );

                let (next_obs, reward) = env.learn_step(action, &mut history, ticker_index);
                buffer.rewards.last_mut().unwrap().clone_from(&reward);
                episode_reward += reward;

                obs = next_obs;
            }

            buffer.compute_advantages();

            // PPO update
            for _ in 0..3 {
                let obs_batch = Tensor::cat(&buffer.observations, 0);
                let actions_batch = Tensor::from_slice(&buffer.actions).view([-1, 1]);
                let old_log_probs = Tensor::from_slice(&buffer.log_probs);
                let advantages = Tensor::from_slice(&buffer.advantages);
                let returns = Tensor::from_slice(&buffer.returns);

                let (action_logits, values) = model.forward(&obs_batch);
                let values = values.squeeze();

                // Check for NaN values and skip update if found
                if values.isnan().any().int64_value(&[]) != 0 {
                    println!("Warning: NaN detected in values, skipping update");
                    break;
                }

                let probs = action_logits.softmax(-1, tch::Kind::Float);
                let log_probs = probs.log().gather(1, &actions_batch, false).squeeze();

                let ratio = (log_probs - old_log_probs).exp();
                let clipped_ratio = ratio.clamp(1.0 - EPSILON, 1.0 + EPSILON);

                let policy_loss =
                    -Tensor::min_other(&(ratio * &advantages), &(clipped_ratio * &advantages))
                        .mean(tch::Kind::Float);

                let value_loss = (values - returns)
                    .pow_tensor_scalar(2)
                    .mean(tch::Kind::Float);

                let entropy = -(probs.copy() * probs.log())
                    .sum_dim_intlist(&[1i64][..], false, tch::Kind::Float)
                    .mean(tch::Kind::Float);

                let total_loss = &policy_loss + VALUE_COEF * &value_loss - ENTROPY_COEF * &entropy;

                // Check for NaN in total loss
                if total_loss.isnan().any().int64_value(&[]) != 0 {
                    println!("Warning: NaN detected in loss, skipping update");
                    break;
                }

                opt.zero_grad();
                total_loss.backward();

                opt.step();
            }

            buffer.clear();
        }
        
        #[cfg(feature = "debug_training")]
        history.record(generation, &mapped_historical);
        
        let (lowest_ticker_index, lowest_assets) = history.ticker_lowest_final_assets();
        let lowest_ticker = TICKERS[lowest_ticker_index as usize];
        let avg_assets = history.avg_final_assets();
        
        println!("generation {} - Lowest assets {} for ticker {}, average {}", generation, lowest_assets, lowest_ticker, avg_assets);
    }

    println!("Training completed!");
}
