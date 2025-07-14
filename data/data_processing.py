import numpy as np
import os
from datetime import datetime
from kitaev_hamiltonian import generate_hamiltonians
from simulate_conductance import generate_conductance_data

def save_run_data(run_dir, params, labels, conductance_maps):
    try:
        os.makedirs(run_dir, exist_ok=True)
        for sample_id, (param, label, conductance_map) in enumerate(zip(params, labels, conductance_maps)):
            if conductance_map is None:
                print(f"Skipping sample {sample_id}: No conductance map")
                continue
            # Save parameters
            with open(os.path.join(run_dir, f"sample_{sample_id}_params.txt"), 'w') as f:
                f.write("t, mu, delta, disorder_strength, local_potential\n")
                f.write(", ".join(map(str, param)) + "\n")
            # Save mu_c (label)
            np.save(os.path.join(run_dir, f"sample_{sample_id}.npy"), label)
    except Exception as e:
        print(f"Error in save_run_data: {e}")

def generate_data():
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = f"/home/levi/mzm_project/data/runs/run_{timestamp}"
        num_samples = 4000  # Increased to 4000
        params, labels, class_labels = generate_hamiltonians(num_samples)
        if params is None or labels is None or class_labels is None:
            print("Failed to generate Hamiltonians")
            return
        conductance_maps = generate_conductance_data(params, class_labels, output_dir=run_dir)
        if conductance_maps is None:
            print("Failed to generate conductance data")
            return
        save_run_data(run_dir, params, labels, conductance_maps)
        print(f"Data generation completed for run: {run_dir}")
    except Exception as e:
        print(f"Error in generate_data: {e}")

if __name__ == "__main__":
    generate_data()