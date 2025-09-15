use rand::{thread_rng, Rng};
use std::fmt::Debug;

pub trait ActionValue: Debug + Copy + Clone {}

impl ActionValue for u32 {}
impl ActionValue for f32 {}

pub trait Action: Debug + Copy + Clone + From<f32> + Into<f32> + From<u32> + Into<u32> {
    fn random() -> Self {
        (thread_rng().gen_range(0..Self::size())).into()
    }

    fn enumerate() -> Vec<Self>;

    fn size() -> usize {
        Self::enumerate().len()
    }
}
