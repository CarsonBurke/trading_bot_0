use rand::{rng, seq::IndexedRandom, Rng};
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};

use crate::{
    constants::TICKERS,
    custom_rl::{
        ActorCritic, GenHistory, MetaHistory, PPOBuffer, TradingEnvironment, ENTROPY_COEF, EPSILON,
        LEARNING_RATE, VALUE_COEF,
    },
    data::historical::get_historical_data,
};

pub fn train() {
    let mapped_historical = get_historical_data();
    
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let model = ActorCritic::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, LEARNING_RATE).unwrap();

    let mut env = TradingEnvironment::new();
    let mut buffer = PPOBuffer::new();
    let mut meta_history = MetaHistory::default();

    for generation in 0..1000 {
        let mut gen_history = GenHistory::new(mapped_historical.len());

        for (ticker_index, bars) in mapped_historical.iter().enumerate() {
            let prices = bars.iter().map(|bar| bar.close).collect::<Vec<f64>>();

            let mut obs = env.reset(prices.clone());
            let mut episode_reward = 0.0;

            // Collect trajectory
            for _ in env.current_step..prices.len() - 1 {
                let (action_logits, value) = tch::no_grad(|| model.forward(&obs));

                let actions: Vec<f64> = action_logits.flatten(0, -1).try_into().unwrap();

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

                let (next_obs, reward) = env.learn_step(&actions, &mut gen_history, ticker_index);
                buffer.rewards.last_mut().unwrap().clone_from(&reward);
                episode_reward += reward;

                obs = next_obs;
            }

            buffer.compute_advantages();

            // PPO update
            for _ in 0..3 {
                let obs_batch = Tensor::cat(&buffer.observations, 0);
                let actions_batch = Tensor::from_slice(&buffer.actions);
                let old_log_probs = Tensor::from_slice(&buffer.log_probs);
                let advantages = Tensor::from_slice(&buffer.advantages);
                let returns = Tensor::from_slice(&buffer.returns);

                let (action_output, values) = model.forward(&obs_batch);
                let values = values.squeeze();

                // Check for NaN values and skip update if found
                if values.isnan().any().int64_value(&[]) != 0 {
                    println!("Warning: NaN detected in values, skipping update");
                    break;
                }

                // For continuous actions (% position), compute log probability assuming Gaussian distribution
                // action_output should contain mean and log_std for the distribution
                let action_mean = action_output.narrow(1, 0, 1).squeeze();
                let action_log_std = action_output.narrow(1, 1, 1).squeeze();
                let action_std = action_log_std.exp();

                // Compute log probabilities for continuous actions
                let log_probs = -0.5 * ((actions_batch - &action_mean) / &action_std).pow_tensor_scalar(2)
                    - action_log_std.copy()
                    - 0.5 * std::f64::consts::LN_2 * std::f64::consts::PI;

                let ratio: Tensor = (&log_probs - &old_log_probs).exp();
                let clipped_ratio = ratio.clamp(1.0 - EPSILON, 1.0 + EPSILON);

                let policy_loss =
                    -Tensor::min_other(&(ratio * &advantages), &(clipped_ratio * &advantages))
                        .mean(tch::Kind::Float);

                let value_loss = (values - returns)
                    .pow_tensor_scalar(2)
                    .mean(tch::Kind::Float);

                // Entropy for continuous Gaussian distribution
                let entropy = 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E)
                    .ln() + action_log_std.mean(tch::Kind::Float);

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
        if generation % 10 == 0 {
            gen_history.record(generation, &mapped_historical);

            meta_history.record(&gen_history);
            meta_history.chart(generation);
        }

        let (lowest_ticker_index, lowest_assets) = gen_history.ticker_lowest_final_assets();
        let lowest_ticker = TICKERS[lowest_ticker_index as usize];
        let avg_assets = gen_history.avg_final_assets();

        println!(
            "generation {} - Lowest assets {} for ticker {}, average {}",
            generation, lowest_assets, lowest_ticker, avg_assets
        );
    }

    println!("Training completed!");
}
