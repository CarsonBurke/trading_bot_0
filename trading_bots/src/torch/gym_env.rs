//! Wrappers around the Python API of the OpenAI gym.
use pyo3::prelude::*;
use tch::Tensor;

/// The return value for a step.
#[derive(Debug)]
pub struct Step<A> {
    pub obs: Tensor,
    pub action: A,
    pub reward: f64,
    pub is_done: bool,
}

impl<A: Copy> Step<A> {
    /// Returns a copy of this step changing the observation tensor.
    pub fn copy_with_obs(&self, obs: &Tensor) -> Step<A> {
        Step {
            obs: obs.copy(),
            action: self.action,
            reward: self.reward,
            is_done: self.is_done,
        }
    }
}

/// An OpenAI Gym session.
pub struct GymEnv {
    env: Py<PyAny>,
    action_space: i64,
    observation_space: Vec<i64>,
}

impl GymEnv {
    /// Creates a new session of the specified OpenAI Gym environment.
    pub fn new(name: &str) -> PyResult<GymEnv> {
        Python::with_gil(|py| {
            let gym = py.import("gym")?;
            let env = gym.call_method1("make", (name,))?;
            let _ = env.call_method1("seed", (42,))?;
            
            let action_space = env.getattr("action_space")?;
            let action_space = if let Ok(val) = action_space.getattr("n") {
                val.extract()?
            } else {
                let action_space: Vec<i64> = action_space.getattr("shape")?.extract()?;
                action_space[0]
            };
            
            let observation_space = env.getattr("observation_space")?;
            let observation_space: Vec<i64> = observation_space.getattr("shape")?.extract()?;
            
            Ok(GymEnv {
                env: env.into(),
                action_space,
                observation_space,
            })
        })
    }

    /// Resets the environment, returning the observation tensor.
    pub fn reset(&self) -> PyResult<Tensor> {
        Python::with_gil(|py| {
            let obs = self.env.call_method0(py, "reset")?;
            let obs_vec: Vec<f32> = obs.extract(py)?;
            Ok(Tensor::from_slice(&obs_vec))
        })
    }

    /// Applies an environment step using the specified action.
    pub fn step<A: IntoPy<Py<PyAny>> + Copy>(&self, action: A) -> PyResult<Step<A>> {
        Python::with_gil(|py| {
            let step = self.env.call_method1(py, "step", (action,))?;
            let step = step.bind(py);
            
            let obs: Vec<f32> = step.get_item(0)?.extract()?;
            let reward: f64 = step.get_item(1)?.extract()?;
            let is_done: bool = step.get_item(2)?.extract()?;
            
            Ok(Step {
                obs: Tensor::from_slice(&obs),
                reward,
                is_done,
                action,
            })
        })
    }

    /// Returns the number of allowed actions for this environment.
    pub fn action_space(&self) -> i64 {
        self.action_space
    }

    /// Returns the shape of the observation tensors.
    pub fn observation_space(&self) -> &[i64] {
        &self.observation_space
    }
}
