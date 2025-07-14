import numpy as np
import os
import logging
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks

class ConductanceDataset(Dataset):
    def __init__(self, conductance_maps, labels, extra_features=None):
        if conductance_maps.ndim == 4 and conductance_maps.shape[1] == 1:
            conductance_maps = conductance_maps.squeeze(1)
        conductance_maps = conductance_maps[:, np.newaxis, :, :]
        self.conductance_maps = torch.tensor(conductance_maps, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.extra_features = extra_features if extra_features is not None else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.extra_features is not None:
            return self.conductance_maps[idx], self.labels[idx], self.extra_features[idx]
        return self.conductance_maps[idx], self.labels[idx]

def extract_fft_features(conductance_maps):
    """Extract FFT-based features from conductance maps."""
    fft_features = []
    for map in conductance_maps:
        fft = np.fft.fft2(map)
        fft_magnitude = np.abs(fft)
        fft_magnitude = fft_magnitude.flatten()
        fft_magnitude[0] = 0  # Exclude DC component
        top_indices = np.argsort(fft_magnitude)[-10:]  # Top 10 frequencies
        top_frequencies = fft_magnitude[top_indices]
        fft_features.append(top_frequencies)
    return np.array(fft_features)

def extract_statistical_features(conductance_maps):
    """Extract statistical moments (mean, variance, skewness) from conductance maps."""
    stats_features = []
    for map in conductance_maps:
        mean = np.mean(map)
        variance = np.var(map)
        skewness = np.mean((map - mean) ** 3) / (np.var(map) ** 1.5 + 1e-8)
        stats_features.append([mean, variance, skewness])
    return np.array(stats_features)

def extract_peak_features(conductance_maps):
    """Extract peak characteristics (number of peaks, average peak height) from conductance maps."""
    peak_features = []
    for map in conductance_maps:
        # Compute peaks along the chemical potential axis (average over voltage bias)
        profile = np.mean(map, axis=0)
        peaks, properties = find_peaks(profile, height=0.1, prominence=0.05)
        num_peaks = len(peaks)
        avg_peak_height = np.mean(properties['peak_heights']) if len(peaks) > 0 else 0.0
        peak_features.append([num_peaks, avg_peak_height])
    return np.array(peak_features)

def load_data(label_dir, image_dir, num_samples=4000, test_size=0.25, batch_size=64, random_seed=42, filename_pattern="conductance_image_{id}.png"):
    try:
        if not os.path.exists(label_dir):
            logging.error(f"Label directory not found: {label_dir}")
            raise FileNotFoundError(f"Label directory not found: {label_dir}")
        if not os.path.exists(image_dir):
            logging.error(f"Image directory not found: {image_dir}")
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        class_labels_path = os.path.join(label_dir, "class_labels.npy")
        if not os.path.exists(class_labels_path):
            logging.error(f"Class labels file not found: {class_labels_path}")
            raise FileNotFoundError(f"Class labels file not found: {class_labels_path}")
        class_labels = np.load(class_labels_path)
        logging.info(f"Class labels shape: {class_labels.shape}, Unique values: {np.unique(class_labels)}")

        binary_labels = np.where(class_labels == 0, 0, 1)
        num_topological = np.sum(binary_labels == 0)
        num_trivial = np.sum(binary_labels == 1)
        logging.info(f"Label distribution: Topological (0): {num_topological}, Trivial (1): {num_trivial}")

        conductance_maps = []
        valid_indices = []
        max_conductance = 1.022036
        for sample_id in range(num_samples):
            map_path = os.path.join(image_dir, filename_pattern.format(id=sample_id))
            if os.path.exists(map_path):
                img = Image.open(map_path).convert('L')
                img = img.resize((30, 30))
                conductance_map = np.array(img, dtype=np.float32)
                conductance_map = (conductance_map / 255.0) * max_conductance
                conductance_maps.append(conductance_map)
                valid_indices.append(sample_id)
            else:
                logging.warning(f"Skipping sample {sample_id}: File missing at {map_path}")

        conductance_maps = np.array(conductance_maps)
        labels = binary_labels[valid_indices]
        logging.info(f"Checked {num_samples} samples, found {len(conductance_maps)} valid conductance maps")
        if len(conductance_maps) == 0:
            logging.error("No valid conductance maps found")
            raise ValueError("No valid conductance maps found")

        # Normalize conductance maps to zero mean and unit variance
        mean = conductance_maps.mean()
        std = conductance_maps.std()
        conductance_maps = (conductance_maps - mean) / (std + 1e-8)

        # Extract multiple types of features
        fft_features = extract_fft_features(conductance_maps)
        stats_features = extract_statistical_features(conductance_maps)
        peak_features = extract_peak_features(conductance_maps)
        extra_features = np.concatenate([fft_features, stats_features, peak_features], axis=1)

        X_train, X_test, y_train, y_test, extra_train, extra_test = train_test_split(
            conductance_maps, labels, extra_features, test_size=test_size, random_state=random_seed
        )

        logging.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

        train_dataset = ConductanceDataset(X_train, y_train, extra_train)
        test_dataset = ConductanceDataset(X_test, y_test, extra_test)

        class_sample_count = np.array([num_topological, num_trivial])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[label] for label in y_train])
        samples_weight = torch.from_numpy(samples_weight).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        logging.info(f"Train dataset: {len(train_dataset)} samples, X_train shape: {X_train.shape}")
        logging.info(f"Test dataset: {len(test_dataset)} samples, X_test shape: {X_test.shape}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader, X_test, y_test, y_train, extra_train, extra_test

    except Exception as e:
        logging.error(f"Error in load_data: {e}")
        return None, None, None, None, None, None, None