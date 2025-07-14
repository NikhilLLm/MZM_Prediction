import numpy as np
import os
import matplotlib.pyplot as plt
import glob  # <-- Add this import

def load_data(run_dir, num_samples):
    params = []
    labels = []
    conductance_maps = []
    for sample_id in range(num_samples):
        # Load parameters
        param_path = os.path.join(run_dir, f"sample_{sample_id}_params.txt")
        with open(param_path, 'r') as f:
            lines = f.readlines()
            param_line = lines[1].strip().split(', ')
            params.append(tuple(float(x) for x in param_line))  # (t, mu, delta, disorder_strength, local_potential)
        # Load mu_c (label)
        label_path = os.path.join(run_dir, f"sample_{sample_id}.npy")
        mu_c = np.load(label_path).item()
        labels.append(mu_c)
        # Load conductance map
        map_path = os.path.join(run_dir, f"sample_{sample_id}_conductance.npy")
        conductance_map = np.load(map_path)
        conductance_maps.append(conductance_map)
    return params, labels, conductance_maps

def save_conductance_as_image(conductance_map, sample_id, output_dir, v_bias_range=(-0.28, 0.28), mu_range=(-60, 60)):
    # Create a figure with a larger size
    plt.figure(figsize=(8, 6), dpi=300)

    # Plot the transposed conductance map with interpolation
    im = plt.imshow(
        conductance_map.T,  # Transpose for axis swap
        cmap='jet',  # Use 'jet' colormap for better contrast
        interpolation='bilinear',  # Smooth the plot
        extent=[mu_range[0], mu_range[1], v_bias_range[1], v_bias_range[0]],  # Swap x/y extents
        aspect='auto'
    )

    # Add labels and title
    plt.xlabel('Chemical Potential \( \mu \) (meV)')
    plt.ylabel('Voltage Bias \( V \) (meV)')
    plt.title(f'Conductance Map (Sample {sample_id})')

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Conductance (\( 2e^2/h \))')

    # Save the image
    output_path = os.path.join(output_dir, f"conductance_image_{sample_id}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved conductance image for sample {sample_id} to {output_path}")

def generate_conductance_images(run_dir, num_samples=4000, output_dir="/home/levi/mzm_project/images"):
    # Create output directory for images
    os.makedirs(output_dir, exist_ok=True)
    # Remove all old images before saving new ones
    for f in glob.glob(os.path.join(output_dir, "conductance_image_*.png")):
        os.remove(f)
    
    # Load data
    params, labels, conductance_maps = load_data(run_dir, num_samples)
    
    # Convert each conductance map to an image
    for sample_id, conductance_map in enumerate(conductance_maps):
        save_conductance_as_image(
            conductance_map,
            sample_id,
            output_dir,
            v_bias_range=(-0.28, 0.28),
            mu_range=(-60, 60)  # Updated to match the new range
        )

if __name__ == "__main__":
    # Replace with your run directory
    run_dir = "/home/levi/mzm_project/data/runs/run_2025-05-08_19-01-08"
    generate_conductance_images(run_dir)