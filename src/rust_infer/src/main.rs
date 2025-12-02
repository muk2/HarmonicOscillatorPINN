use anyhow::Result;
use tch::{CModule, Device, Kind, Tensor};

fn main() -> Result<()> {
    // Define the path to your saved TorchScript model
    let model_path = "../python/pinn_harmonic.pt";

    // Load the model
    let model = CModule::load(model_path)?;
    println!("Successfully loaded model: {}", model_path);

    // --- Prepare input data for testing (equivalent to t_test in Python) ---
    let device = Device::Cpu;
    let start_time = 0.0;
    let end_time = 6.2831853; // 2 * PI
    let num_test_points = 400;

    let times = Tensor::linspace(start_time, end_time, num_test_points, (Kind::Float, device));

    // Reshape to [400, 1]
    let input_tensor = times.view([-1, 1]);

    // --- Run Inference ---
    let output_tensor = model.forward_ts(&[input_tensor])?;

    // --- Process Output ---

    // FIX 1: Make the output tensor contiguous/flattened before conversion
    let output_data: Vec<f32> = output_tensor.flatten(0, 1).try_into()?;

    // We can flatten 'times' as well if we didn't already use it 1D
    let times_data: Vec<f32> = times.flatten(0, 0).try_into()?;


    println!("Inference successful. First 5 predictions:");
    for i in 0..5 {
        println!("  t={:.4}, x_pred={:.4}", times_data[i], output_data[i]);
    }

    // You now have `output_data` which corresponds to `y` in your Python script.

    Ok(())
}
