import numpy as np
import os
from datetime import datetime
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
from dataloader import load_data
from model import ConductanceCNN
from train import train_model, save_model
from tda_features import compute_tda_features
import logging

# Set up logging
logging.basicConfig(filename='/home/levi/mzm_project/results/train.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_gradcam_heatmap(model, img, class_idx, device='cuda'):
    model.eval()
    img = img.to(device)
    model = model.to(device)
    
    # Forward pass to get conv features and output
    img.requires_grad_(True)
    conv_features = model.get_conv_features(img)
    output, _ = model(img)
    output = output[0, class_idx]
    
    # Backward pass to get gradients
    model.zero_grad()
    output.backward()
    grads = img.grad[0]  # Gradients of the input image
    pooled_grads = torch.mean(grads, dim=[1, 2], keepdim=True)
    
    # Compute heatmap
    heatmap = torch.mean(conv_features * pooled_grads, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    return heatmap.cpu().detach().numpy()

def run_pipeline(run_dir, num_samples=4000, num_epochs=10, device='cuda'):
    try:
        # Create results directory with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_dir = f"/home/levi/mzm_project/results/run_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "gradcam"), exist_ok=True)

        # Update logging to write to the new results directory
        logging.getLogger().handlers[0].baseFilename = os.path.join(results_dir, "train.log")

        # Load data
        logging.info("Loading data...")
        train_loader, val_loader, test_loader, X_test, y_test, y_train = load_data(run_dir, num_samples)
        if train_loader is None:
            raise ValueError("Failed to load data")

        # Train CNN
        logging.info("Training CNN...")
        model = ConductanceCNN()
        cnn_metrics = train_model(model, train_loader, val_loader, test_loader, y_train, num_epochs, device=device)
        if cnn_metrics[0] is None:
            raise ValueError("Failed to train CNN")
        cnn_accuracy, cnn_precision, cnn_recall, cnn_f1, cnn_pred, y_test = cnn_metrics

        # Save model
        model_path = os.path.join(results_dir, "model.pth")
        save_model(model, model_path)

        # Compute TDA features and classify with Random Forest
        logging.info("Computing TDA features...")
        # Combine X_test and part of the training data for TDA (to match split in compute_tda_features)
        X_all = np.concatenate([X_test, X_train[:len(X_train)//2]], axis=0)
        y_all = np.concatenate([y_test, y_train[:len(y_train)//2]], axis=0)
        tda_results = compute_tda_features(X_all, y_all, results_dir, samples_to_visualize=[0, 1, len(X_test), len(X_test)+1])
        if tda_results[0] is None:
            raise ValueError("Failed to compute TDA features")
        tda_features, tda_accuracy, tda_precision, tda_recall, tda_f1, tda_pred, tda_y_test = tda_results

        # Save performance metrics (combined CNN and TDA+RF)
        logging.info("Saving performance metrics...")
        with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
            f.write("Performance Metrics (Test Set):\n")
            f.write("Model\tAccuracy\tPrecision (Topo)\tRecall (Topo)\tF1 (Topo)\tPrecision (Triv)\tRecall (Triv)\tF1 (Triv)\n")
            f.write(f"CNN\t{cnn_accuracy:.3f}\t\t{cnn_precision[0]:.3f}\t\t{cnn_recall[0]:.3f}\t\t{cnn_f1[0]:.3f}\t\t{cnn_precision[1]:.3f}\t\t{cnn_recall[1]:.3f}\t\t{cnn_f1[1]:.3f}\n")
            f.write(f"TDA+RF\t{tda_accuracy:.3f}\t\t{tda_precision[0]:.3f}\t\t{tda_recall[0]:.3f}\t\t{tda_f1[0]:.3f}\t\t{tda_precision[1]:.3f}\t\t{tda_recall[1]:.3f}\t\t{tda_f1[1]:.3f}\n")

        # Confusion Matrix (CNN)
        logging.info("Generating confusion matrix for CNN...")
        cm = confusion_matrix(y_test, cnn_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Topological', 'Trivial'], yticklabels=['Topological', 'Trivial'])
        plt.title('Confusion Matrix (CNN - Test Set)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
        plt.close()

        # PCA Visualization (CNN Features)
        logging.info("Generating PCA scatter plot...")
        model.eval()
        test_features = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                _, features = model(inputs)
                test_features.append(features.cpu().numpy())
        test_features = np.concatenate(test_features, axis=0)
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(test_features)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=y_test, cmap='coolwarm', alpha=0.6)
        plt.colorbar(scatter, ticks=[0, 1], label='Class')
        plt.clim(-0.5, 1.5)
        plt.gca().set_xticks([])  # Remove x-axis ticks
        plt.gca().set_yticks([])  # Remove y-axis ticks
        plt.title('PCA of CNN Features (Test Set)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(handles=scatter.legend_elements()[0], labels=['Topological', 'Trivial'], loc='best')
        plt.savefig(os.path.join(results_dir, "pca_scatter.png"))
        plt.close()

        # Grad-CAM Visualization
        logging.info("Generating Grad-CAM visualizations...")
        model.eval()
        # Select samples: 2 topological, 2 trivial
        topo_indices = np.where(y_test == 0)[0][:2]
        triv_indices = np.where(y_test == 1)[0][:2]
        selected_indices = np.concatenate([topo_indices, triv_indices])
        
        for idx in selected_indices:
            img = torch.tensor(X_test[idx:idx+1], dtype=torch.float32).unsqueeze(-1)  # Add channel dimension
            true_label = y_test[idx]
            pred_label = cnn_pred[idx]
            heatmap = get_gradcam_heatmap(model, img, pred_label, device)
            
            # Plot conductance map with heatmap overlay
            plt.figure(figsize=(8, 6))
            plt.imshow(img[0, :, :, 0].numpy(), cmap='jet', extent=[-60, 60, 0.28, -0.28], aspect='auto')
            plt.imshow(heatmap, cmap='jet', alpha=0.5, extent=[-60, 60, 0.28, -0.28], aspect='auto')
            plt.colorbar(label='Conductance (2e²/h)')
            plt.xlabel('Chemical Potential (μ, meV)')
            plt.ylabel('Voltage Bias (V, meV)')
            plt.title(f'Grad-CAM: True={["Topological", "Trivial"][true_label]}, Pred={["Topological", "Trivial"][pred_label]}')
            plt.savefig(os.path.join(results_dir, f"gradcam/sample_{idx}.png"))
            plt.close()

        logging.info("Pipeline completed successfully")

    except Exception as e:
        logging.error(f"Error in run_pipeline: {e}")

if __name__ == "__main__":
    run_dir = "/home/levi/mzm_project/data/runs/run_<timestamp>"  # Update with your timestamp
    run_pipeline(run_dir, num_samples=4000)