// Vectorized version of the gym environment.
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use tch::Tensor;

#[derive(Debug)]
pub struct Step {
    pub obs: Tensor,
    pub reward: Tensor,
    pub is_done: Tensor,
}

pub struct VecGymEnv {
    env: Py<PyAny>,
    pub action_space: i64,
    pub observation_space: Vec<i64>,
}

impl VecGymEnv {
    pub fn new(name: &str, img_dir: Option<&str>, nprocesses: i64) -> PyResult<VecGymEnv> {
        Python::with_gil(|py| {
            let sys = py.import("sys")?;
            let path = sys.getattr("path")?;
            path.call_method1("append", ("examples/reinforcement-learning",))?;
            
            let gym = py.import("atari_wrappers")?;
            let env = gym.call_method1("make", (name, img_dir, nprocesses))?;
            
            let action_space = env.getattr("action_space")?;
            let action_space: i64 = action_space.getattr("n")?.extract()?;
            
            let observation_space = env.getattr("observation_space")?;
            let observation_space: Vec<i64> = observation_space.getattr("shape")?.extract()?;
            let observation_space = [vec![nprocesses].as_slice(), observation_space.as_slice()].concat();
            
            Ok(VecGymEnv {
                env: env.into(),
                action_space,
                observation_space,
            })
        })
    }

    pub fn reset(&self) -> PyResult<Tensor> {
        Python::with_gil(|py| {
            let obs = self.env.call_method0(py, "reset")?;
            let obs = obs.bind(py);
            let obs = obs.call_method0("flatten")?;
            let obs_vec: Vec<f32> = obs.extract()?;
            Ok(Tensor::from_slice(&obs_vec).view_(&self.observation_space))
        })
    }

    pub fn step(&self, action: Vec<i64>) -> PyResult<Step> {
        Python::with_gil(|py| {
            let step = self.env.call_method1(py, "step", (action,))?;
            let step = step.bind(py);
            
            let obs = step.get_item(0)?;
            let obs = obs.call_method0("flatten")?;
            let obs_buffer = PyBuffer::get(&obs)?;
            let obs_vec: Vec<u8> = obs_buffer.to_vec(py)?;
            let obs = Tensor::from_slice(&obs_vec)
                .view_(&self.observation_space)
                .to_kind(tch::Kind::Float);
            
            let reward_vec: Vec<f32> = step.get_item(1)?.extract()?;
            let reward = Tensor::from_slice(&reward_vec);
            
            let is_done_vec: Vec<f32> = step.get_item(2)?.extract()?;
            let is_done = Tensor::from_slice(&is_done_vec);
            
            Ok(Step { obs, reward, is_done })
        })
    }

    pub fn action_space(&self) -> i64 {
        self.action_space
    }

    pub fn observation_space(&self) -> &[i64] {
        &self.observation_space
    }
}
