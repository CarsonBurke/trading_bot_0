use rand::Rng;

#[derive(Debug, Clone)]
pub struct ActionsContinuous(pub Vec<f32>);

impl ActionsContinuous {
    pub fn new(values: Vec<f32>) -> Self {
        ActionsContinuous(values)
    }
    
    pub fn new_random(size: usize) -> Self {
        let mut rng = rand::rng();
        let values = (0..size).map(|_| rng.random_range(-1.0..1.0)).collect();
        ActionsContinuous(values)
    }
    
    pub fn new_random_with_range(size: usize, min: f32, max: f32) -> Self {
        let mut rng = rand::rng();
        let values = (0..size).map(|_| rng.random_range(min..max)).collect();
        ActionsContinuous(values)
    }
    
    pub fn size(&self) -> usize {
        self.0.len()
    }
}