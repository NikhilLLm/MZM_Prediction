import numpy as np
import os
import shutil
from datetime import datetime
from kitaev_hamiltonian import generate_hamiltonians
from simulate_conductance import generate_conductance_data
import matplotlib.pyplot as plt
import glob
import logging

# Set up logging
logging.basicConfig(filename='/home/levi/mzm_project/run.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Data Generation
def save_run_data(run_dir, params, labels, conductance_maps):
    try:
        os.makedirs(run_dir, exist_ok=True)
        successful_samples = 0
        for sample_id, (param, label, conductance_map) in enumerate(zip(params, labels, conductance_maps)):
            # Skip if conductance map is None or invalid
            if conductance_map is None or not isinstance(conductance_map, np.ndarray) or conductance_map.shape != (30, 30):
                logging.warning(f"Skipping sample {sample_id}: Invalid or missing conductance map")
                continue
            # Save parameters
            with open(os.path.join(run_dir, f"sample_{sample_id}_params.txt"), 'w') as f:
                f.write("t, mu, delta, disorder_strength, local_potential\n")
                f.write(", ".join(map(str, param)) + "\n")
            # Save mu_c (label)
            np.save(os.path.join(run_dir, f"sample_{sample_id}.npy"), label)
            successful_samples += 1
        logging.info(f"Saved data for {successful_samples} samples in {run_dir}")
    except Exception as e:
        logging.error(f"Error in save_run_data: {e}")
        raise

# Step 2: Image Generation
def save_conductance_as_image(conductance_map, sample_id, output_dir, v_bias_range=(-0.28, 0.28), mu_range=(-60, 60)):
    try:
        plt.figure(figsize=(8, 6), dpi=300)
        im = plt.imshow(
            conductance_map.T,
            cmap='jet',
            interpolation='bilinear',
            extent=[mu_range[0], mu_range[1], v_bias_range[1], v_bias_range[0]],
            aspect='auto'
        )
        plt.xlabel('Chemical Potential \( \mu \) (meV)')
        plt.ylabel('Voltage Bias \( V \) (meV)')
        plt.title(f'Conductance Map (Sample {sample_id})')
        cbar = plt.colorbar(im)
        cbar.set_label('Conductance (\( 2e^2/h \))')
        output_path = os.path.join(output_dir, f"conductance_image_{sample_id}.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved conductance image for sample {sample_id} to {output_path}")
    except Exception as e:
        logging.error(f"Error in save_conductance_as_image for sample {sample_id}: {e}")

# Combined Main Function
def generate_all_data(num_samples=4000, output_base_dir="/home/levi/mzm_project"):
    try:
        # Create run directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = f"{output_base_dir}/data/runs/run_{timestamp}"
        image_dir = f"{output_base_dir}/images"
        
        # Log start of the run
        logging.info(f"Starting data generation in {run_dir}")

        # Step 1: Generate Hamiltonians
        logging.info("Generating Hamiltonians...")
        params, labels, class_labels = generate_hamiltonians(num_samples)
        if params is None or labels is None or class_labels is None:
            logging.error("Failed to generate Hamiltonians")
            raise ValueError("Failed to generate Hamiltonians")
        
        # Step 2: Generate Conductance Maps
        logging.info("Generating conductance data...")
        conductance_maps = generate_conductance_data(params, class_labels, output_dir=run_dir)
        if conductance_maps is None:
            logging.error("Failed to generate conductance data")
            raise ValueError("Failed to generate conductance data")
        
        # Step 3: Save Data (only valid samples)
        logging.info("Saving run data...")
        save_run_data(run_dir, params, labels, conductance_maps)

        # Step 4: Generate Images
        logging.info(f"Generating images in {image_dir}...")
        os.makedirs(image_dir, exist_ok=True)
        for f in glob.glob(os.path.join(image_dir, "conductance_image_*.png")):
            os.remove(f)
        
        successful_images = 0
        for sample_id, conductance_map in enumerate(conductance_maps):
            if conductance_map is not None and isinstance(conductance_map, np.ndarray) and conductance_map.shape == (30, 30):
                save_conductance_as_image(
                    conductance_map,
                    sample_id,
                    image_dir,
                    v_bias_range=(-0.28, 0.28),
                    mu_range=(-60, 60)
                )
                successful_images += 1
        logging.info(f"Generated {successful_images}/{num_samples} images")

        # Validate the run
        if successful_images < num_samples * 0.9:  # If less than 90% of samples succeeded
            logging.error("Too many failed samples, cleaning up...")
            shutil.rmtree(run_dir, ignore_errors=True)
            shutil.rmtree(image_dir, ignore_errors=True)
            raise RuntimeError("Run failed: Too many samples failed to generate")

        logging.info("Data generation and image creation completed successfully")

    except Exception as e:
        # Clean up if an error occurs
        logging.error(f"Error in generate_all_data: {e}")
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir, ignore_errors=True)
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir, ignore_errors=True)
        raise

if __name__ == "__main__":
    generate_all_data(num_samples=4000)