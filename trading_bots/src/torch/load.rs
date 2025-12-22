use std::error::Error;
use std::path::Path;
use tch::nn;

pub fn load_var_store_partial<P: AsRef<Path>>(
    vs: &mut nn::VarStore,
    path: P,
) -> Result<Vec<String>, Box<dyn Error>> {
    let missing = vs.load_partial(path)?;
    if !missing.is_empty() {
        println!("Loaded weights with {} missing tensors", missing.len());
    }
    Ok(missing)
}
