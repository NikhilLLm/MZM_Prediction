import numpy as np
import cupy as cp
import kwant
import os
from multiprocessing import Pool
from kitaev_hamiltonian import kitaev_hamiltonian
import logging

# Set up logging
logging.basicConfig(filename='/home/levi/mzm_project/run.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def make_kitaev_system(t, mu, delta, disorder_strength, local_potential, L):
    try:
        lat = kwant.lattice.chain(norbs=2)  # Spinless: 2 for particle-hole
        syst = kwant.Builder()
        disorder_arr = np.random.normal(0, 1, L) * disorder_strength if disorder_strength > 0 else np.zeros(L)
        local_arr = np.array([
            local_potential * np.exp(-((i-0)**2 + (i-(L-1))**2) / 10) if local_potential > 0 else 0
            for i in range(L)
        ])
        for i in range(L):
            syst[lat(i)] = (lambda i: (lambda site, mu: np.array([
                [-mu + disorder_arr[i] + local_arr[i], 0],
                [0, mu - disorder_arr[i] - local_arr[i]]
            ])))(i)
        for i in range(L-1):
            hopping = np.array([[-t, delta], [-delta, t]])
            syst[lat(i), lat(i+1)] = hopping
        lead = kwant.Builder(kwant.TranslationalSymmetry([-1]))
        lead[lat(0)] = lambda site, mu_lead: np.array([[0, 0], [0, 0]])
        lead[lat(0), lat(1)] = np.array([[-t, 0], [0, t]])
        syst.attach_lead(lead)
        syst.attach_lead(lead.reversed())
        return syst
    except Exception as e:
        logging.error(f"Error in make_kitaev_system: {e}")
        return None

def compute_conductance(syst, v_bias, mu_values):
    try:
        syst = syst.finalized()
        conductance = np.zeros((len(mu_values), len(v_bias)))
        for i, mu in enumerate(mu_values):
            for j, vb in enumerate(v_bias):
                try:
                    smatrix = kwant.smatrix(syst, energy=vb, params={'mu': mu, 'mu_lead': 0.0})
                    n_modes_left = smatrix.num_propagating(0)
                    n_modes_right = smatrix.num_propagating(1)
                    if n_modes_left == 0 or n_modes_right == 0:
                        logging.warning(f"No propagating modes (mu={mu}, vb={vb}): Left={n_modes_left}, Right={n_modes_right}")
                        conductance[i, j] = 0
                        continue
                    r_ee = smatrix.transmission(1, 0)
                    conductance[i, j] = max(r_ee, 0)
                except Exception as e:
                    logging.error(f"Error in smatrix computation (mu={mu}, vb={vb}): {e}")
                    conductance[i, j] = 0
        return conductance
    except Exception as e:
        logging.error(f"Error in compute_conductance: {e}")
        return None

def process_sample(args):
    i, t, mu, delta, disorder, local, L, v_bias, mu_values, output_dir = args
    try:
        logging.info(f"Processing sample {i} with t={t}, mu={mu}, delta={delta}, disorder={disorder}, local={local}")
        syst = make_kitaev_system(t, mu, delta, disorder, local, L)
        if syst is None:
            logging.error(f"Sample {i}: Failed to create system")
            return None
        conductance = compute_conductance(syst, v_bias, mu_values)
        if conductance is None:
            logging.error(f"Sample {i}: Failed to compute conductance")
            return None
        np.save(os.path.join(output_dir, f"sample_{i}_conductance.npy"), conductance)
        np.savetxt(os.path.join(output_dir, f"sample_{i}_conductance_readable.txt"), conductance, fmt='%.6f')
        logging.info(f"Sample {i}: Successfully computed conductance map")
        return i, conductance
    except Exception as e:
        logging.error(f"Error in sample {i}: {e}")
        return None

def generate_conductance_data(params, class_labels, L=200, output_dir="/home/levi/mzm_project/data"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        v_bias = cp.linspace(-0.28, 0.28, 30).get()  # Reduced to 30 points
        mu_values = cp.linspace(-60, 60, 30).get()  # Reduced to 30 points
        args = [(i, t, mu, delta, disorder, local, L, v_bias, mu_values, output_dir)
                for i, (t, mu, delta, disorder, local) in enumerate(params)]
        with Pool() as pool:
            results = pool.map(process_sample, args)
        conductance_maps = [None] * len(params)
        successful_samples = 0
        for result in results:
            if result is not None:
                i, conductance = result
                conductance_maps[i] = conductance
                successful_samples += 1
        # Save class labels
        np.save(os.path.join(output_dir, "class_labels.npy"), np.array(class_labels))
        logging.info(f"Generated {successful_samples}/{len(params)} samples in {output_dir}")
        logging.info("Data generation completed")
        return conductance_maps
    except Exception as e:
        logging.error(f"Error in generate_conductance_data: {e}")
        return None

if __name__ == "__main__":
    print("This script should be called from data_processing.py")