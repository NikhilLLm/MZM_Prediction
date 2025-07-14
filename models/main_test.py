import numpy as np
import os
from datetime import datetime
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
import seaborn as sns
from dataloader import load_data
from model import ConductanceResNet
from train import train_model
import logging
from scipy.ndimage import gaussian_filter, zoom
from scipy.stats import gaussian_kde
import scipy.signal

try:
    import scipy.stats
    import sklearn.inspection
except ImportError as e:
    logging.error(f"Missing dependency: {e}. Please install required packages (e.g., pip install scipy scikit-learn)")
    raise e

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'figure.facecolor': 'none',
    'savefig.facecolor': 'none'
})

try:
    plt.style.use('ggplot')
except OSError:
    plt.style.use('default')
    logging.warning("Matplotlib style 'ggplot' not available, using default style.")

def get_gradcam_heatmap(model, img, class_idx, device='cuda'):
    try:
        model.eval()
        img = img.to(device)
        model = model.to(device)
        
        if img.ndim == 3:
            img = img.unsqueeze(1)
        elif img.ndim == 5:
            img = img.squeeze(2)
        if img.ndim != 4:
            raise ValueError(f"Expected 4D input, got {img.ndim}D input with shape {img.shape}")
        
        img = img.requires_grad_(True)
        conv_features = model.get_conv_features(img)
        conv_features.retain_grad()
        x = conv_features
        x = model.resnet.avgpool(x)
        x = model.flatten(x)
        output = model.resnet.fc(x)
        output = output[0, class_idx]
        
        model.zero_grad()
        output.backward()
        grads = conv_features.grad
        if grads is None:
            raise ValueError("Gradients for conv features are None. Ensure the model supports gradient computation.")
        pooled_grads = torch.mean(grads, dim=[0, 2, 3])
        
        heatmap = torch.zeros(conv_features.shape[2:], device=device)
        for i in range(pooled_grads.shape[0]):
            heatmap += pooled_grads[i] * conv_features[0, i, :, :]
        heatmap = F.relu(heatmap)
        max_heatmap = torch.max(heatmap)
        heatmap = heatmap / (max_heatmap + 1e-8)
        heatmap = heatmap.cpu().detach().numpy()
        heatmap = zoom(heatmap, 30/3, order=1)
        heatmap = gaussian_filter(heatmap, sigma=1)
        logging.info(f"Heatmap shape after processing: {heatmap.shape}")
        return heatmap
    except Exception as e:
        logging.error(f"Error in get_gradcam_heatmap: {e}")
        raise e

def optimize_ensemble_weights(cnn_probs, svm_probs, y_true, weight_range=np.linspace(0, 1, 11)):
    try:
        best_accuracy = 0
        best_w_cnn = 0
        for w_cnn in weight_range:
            w_cnn = max(w_cnn, 0.2)
            w_svm = 1 - w_cnn
            ensemble_probs = w_cnn * cnn_probs + w_svm * svm_probs
            ensemble_pred = np.argmax(ensemble_probs, axis=1)
            accuracy = accuracy_score(y_true, ensemble_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_w_cnn = w_cnn
        return best_w_cnn, 1 - best_w_cnn, best_accuracy
    except Exception as e:
        logging.error(f"Error in optimize_ensemble_weights: {e}")
        raise e

def run_pipeline(label_dir, image_dir, num_samples=4000, num_epochs=75, device='cuda', filename_pattern="conductance_image_{id}.png"):
    try:
        results_dir = "/home/levi/mzm_project/results"
        gradcam_dir = os.path.join(results_dir, "gradcam")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(gradcam_dir, exist_ok=True)

        logging.basicConfig(
            filename=os.path.join(results_dir, "train.log"),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='a',
            force=True
        )
        logging.info(f"===== Starting Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")

        logging.info("Loading data...")
        train_loader, test_loader, X_test, y_test, y_train, extra_train, extra_test = load_data(
            label_dir, image_dir, num_samples, filename_pattern=filename_pattern
        )
        if train_loader is None or test_loader is None:
            raise ValueError("Failed to load data")

        # Add data augmentation to the training data
        logging.info("Applying data augmentation to training data...")
        augmented_train_inputs = []
        augmented_train_labels = []
        augmented_train_extra = []
        augmented_y_train = []
        for idx, (inputs, labels, extra) in enumerate(train_loader.dataset):
            if isinstance(inputs, np.ndarray):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            if isinstance(extra, np.ndarray):
                extra = torch.tensor(extra, dtype=torch.float32)
            if not isinstance(inputs, torch.Tensor) or not isinstance(extra, torch.Tensor):
                logging.error(f"Dataset item {idx}: inputs type {type(inputs)}, extra type {type(extra)}")
                raise TypeError(f"Expected torch.Tensor, got inputs: {type(inputs)}, extra: {type(extra)}")

            label_value = labels.item() if isinstance(labels, torch.Tensor) else labels

            # Original sample
            augmented_train_inputs.append(inputs)
            augmented_train_labels.append(label_value)
            augmented_train_extra.append(extra)
            augmented_y_train.append(label_value)
            
            # Add random noise
            noise = torch.randn_like(inputs) * 0.1
            augmented_train_inputs.append(inputs + noise)
            augmented_train_labels.append(label_value)
            augmented_train_extra.append(extra)
            augmented_y_train.append(label_value)
            
            # Horizontal flip
            flipped = torch.flip(inputs, dims=[2])
            augmented_train_inputs.append(flipped)
            augmented_train_labels.append(label_value)
            augmented_train_extra.append(extra)
            augmented_y_train.append(label_value)

        logging.info("Checking types before stacking...")
        for idx, (inp, lbl, ext) in enumerate(zip(augmented_train_inputs, augmented_train_labels, augmented_train_extra)):
            if not isinstance(inp, torch.Tensor):
                logging.error(f"augmented_train_inputs[{idx}] is {type(inp)}")
            if not isinstance(ext, torch.Tensor):
                logging.error(f"augmented_train_extra[{idx}] is {type(ext)}")

        augmented_train_inputs = torch.stack(augmented_train_inputs)
        augmented_train_labels = torch.tensor(augmented_train_labels, dtype=torch.long)
        augmented_train_extra = torch.stack(augmented_train_extra)
        augmented_dataset = torch.utils.data.TensorDataset(augmented_train_inputs, augmented_train_labels, augmented_train_extra)
        train_loader = torch.utils.data.DataLoader(augmented_dataset, batch_size=train_loader.batch_size, shuffle=True)

        y_train = np.array(augmented_y_train)
        logging.info(f"Updated y_train shape: {y_train.shape}")

        logging.info(f"Training ResNet with k-fold cross-validation for {num_epochs} epochs...")
        model = ConductanceResNet()
        cnn_metrics = train_model(model, train_loader, test_loader, y_train, num_epochs=num_epochs, device=device)
        if cnn_metrics is None or cnn_metrics[0] is None:
            raise ValueError("Failed to train ResNet")
        cnn_accuracy, cnn_precision, cnn_recall, cnn_f1, cnn_pred, y_test = cnn_metrics
        logging.info(f"CNN Test Accuracy: {cnn_accuracy:.3f}")
        logging.info(f"CNN Precision: {cnn_precision}")
        logging.info(f"CNN Recall: {cnn_recall}")
        logging.info(f"CNN F1: {cnn_f1}")

        logging.info("Extracting features for SVM...")
        model.eval()
        train_features = []
        test_features = []
        cnn_probs_train = []
        cnn_probs_test = []
        with torch.no_grad():
            for batch_idx, (inputs, _, extra) in enumerate(train_loader):
                try:
                    inputs = inputs.to(device)
                    output, features = model(inputs)
                    probs = F.softmax(output, dim=1).cpu().numpy()
                    combined_features = np.concatenate([features.cpu().numpy(), extra.numpy()], axis=1)
                    train_features.append(combined_features)
                    cnn_probs_train.append(probs)
                except Exception as e:
                    logging.error(f"Error in extracting train features, batch {batch_idx}: {e}")
                    raise e
            for batch_idx, (inputs, _, extra) in enumerate(test_loader):
                try:
                    inputs = inputs.to(device)
                    output, features = model(inputs)
                    probs = F.softmax(output, dim=1).cpu().numpy()
                    combined_features = np.concatenate([features.cpu().numpy(), extra.numpy()], axis=1)
                    test_features.append(combined_features)
                    cnn_probs_test.append(probs)
                except Exception as e:
                    logging.error(f"Error in extracting test features, batch {batch_idx}: {e}")
                    raise e
        train_features = np.concatenate(train_features, axis=0)
        test_features = np.concatenate(test_features, axis=0)
        cnn_probs_train = np.concatenate(cnn_probs_train, axis=0)
        cnn_probs_test = np.concatenate(cnn_probs_test, axis=0)
        logging.info(f"Train features shape: {train_features.shape}, Test features shape: {test_features.shape}")

        logging.info("Training SVM with hyperparameter tuning...")
        class_weights = {0: len(y_train) / (2 * np.sum(y_train == 0)), 1: len(y_train) / (2 * np.sum(y_train == 1))}
        logging.info(f"SVM Class Weights: {class_weights}")
        param_grid = {'C': [0.01, 0.1], 'gamma': ['scale', 0.001]}  # Increased regularization
        svm = GridSearchCV(SVC(kernel='rbf', class_weight=class_weights, random_state=42, probability=True), 
                          param_grid, cv=5, n_jobs=-1)
        svm.fit(train_features, y_train)
        logging.info("SVM training completed successfully")
        logging.info(f"Best SVM parameters: {svm.best_params_}")
        svm_pred = svm.predict(test_features)
        svm_probs = svm.predict_proba(test_features)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        svm_precision = precision_score(y_test, svm_pred, average=None, zero_division=0)
        svm_recall = recall_score(y_test, svm_pred, average=None, zero_division=0)
        svm_f1 = f1_score(y_test, svm_pred, average=None, zero_division=0)
        logging.info(f"SVM Test Accuracy: {svm_accuracy:.3f}")
        logging.info(f"SVM Precision: {svm_precision}")
        logging.info(f"SVM Recall: {svm_recall}")
        logging.info(f"SVM F1: {svm_f1}")

        logging.info("Optimizing ensemble weights...")
        w_cnn, w_svm, val_accuracy = optimize_ensemble_weights(cnn_probs_train, svm.predict_proba(train_features), y_train)
        logging.info(f"Optimal ensemble weights - CNN: {w_cnn:.2f}, SVM: {w_svm:.2f}, Validation Accuracy: {val_accuracy:.3f}")
        
        ensemble_probs = w_cnn * cnn_probs_test + w_svm * svm_probs
        ensemble_pred = np.argmax(ensemble_probs, axis=1)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_precision = precision_score(y_test, ensemble_pred, average=None, zero_division=0)
        ensemble_recall = recall_score(y_test, ensemble_pred, average=None, zero_division=0)
        ensemble_f1 = f1_score(y_test, ensemble_pred, average=None, zero_division=0)
        logging.info(f"Ensemble Test Accuracy: {ensemble_accuracy:.3f}")
        logging.info(f"Ensemble Precision: {ensemble_precision}")
        logging.info(f"Ensemble Recall: {ensemble_recall}")
        logging.info(f"Ensemble F1: {ensemble_f1}")

        if ensemble_accuracy < 0.70:
            logging.warning(f"Ensemble accuracy {ensemble_accuracy:.3f} is below 70%. Consider increasing num_epochs or adjusting FocalLoss parameters.")

        logging.info("Analyzing SVM support vectors...")
        try:
            support_indices = svm.best_estimator_.support_
            support_labels = y_train[support_indices]
            logging.info(f"Number of support vectors: {len(support_indices)}")
            logging.info(f"Support vectors distribution: Topological: {np.sum(support_labels == 0)}, Trivial: {np.sum(support_labels == 1)}")
        except AttributeError as e:
            logging.error(f"Error accessing SVM support vectors: {e}")
            support_indices = []
            support_labels = []

        logging.info("Computing feature importance...")
        feature_names = [f"CNN_{i}" for i in range(512)] + [f"FFT_{i}" for i in range(10)] + \
                        ["Mean", "Variance", "Skewness", "Num_Peaks", "Avg_Peak_Height"]
        perm_importance = permutation_importance(svm.best_estimator_, test_features, y_test, n_repeats=1, random_state=42)
        feature_importance = perm_importance.importances_mean

        # Create a unified results figure
        fig = plt.figure(figsize=(18, 24))
        gs = fig.add_gridspec(6, 4, wspace=0.3, hspace=0.5)

        # 1. Normalized Confusion Matrices (CNN, SVM, Ensemble)
        ax_cm_cnn = fig.add_subplot(gs[0, 0])
        cm_cnn = confusion_matrix(y_test, cnn_pred, normalize='true')
        sns.heatmap(cm_cnn, annot=True, fmt='.2f', cmap='Blues', ax=ax_cm_cnn,
                    xticklabels=['Topological', 'Trivial'], yticklabels=['Topological', 'Trivial'], 
                    cbar_kws={'label': 'Fraction'})
        ax_cm_cnn.set_title('(a) Confusion Matrix (ResNet)')
        ax_cm_cnn.set_xlabel('Predicted')
        ax_cm_cnn.set_ylabel('True')

        ax_cm_svm = fig.add_subplot(gs[0, 1])
        cm_svm = confusion_matrix(y_test, svm_pred, normalize='true')
        sns.heatmap(cm_svm, annot=True, fmt='.2f', cmap='Blues', ax=ax_cm_svm,
                    xticklabels=['Topological', 'Trivial'], yticklabels=['Topological', 'Trivial'],
                    cbar_kws={'label': 'Fraction'})
        ax_cm_svm.set_title('(b) Confusion Matrix (SVM)')
        ax_cm_svm.set_xlabel('Predicted')
        ax_cm_svm.set_ylabel('True')

        ax_cm_ensemble = fig.add_subplot(gs[0, 2])
        cm_ensemble = confusion_matrix(y_test, ensemble_pred, normalize='true')
        sns.heatmap(cm_ensemble, annot=True, fmt='.2f', cmap='Blues', ax=ax_cm_ensemble,
                    xticklabels=['Topological', 'Trivial'], yticklabels=['Topological', 'Trivial'],
                    cbar_kws={'label': 'Fraction'})
        ax_cm_ensemble.set_title('(c) Confusion Matrix (Ensemble)')
        ax_cm_ensemble.set_xlabel('Predicted')
        ax_cm_ensemble.set_ylabel('True')

        # 2. ROC Curve for SVM and Ensemble
        ax_roc = fig.add_subplot(gs[0, 3])
        fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probs[:, 1])
        roc_auc_svm = auc(fpr_svm, tpr_svm)
        ax_roc.plot(fpr_svm, tpr_svm, color='darkorange', lw=2, label=f'SVM (AUC = {roc_auc_svm:.2f})')
        
        fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, ensemble_probs[:, 1])
        roc_auc_ensemble = auc(fpr_ensemble, tpr_ensemble)
        ax_roc.plot(fpr_ensemble, tpr_ensemble, color='green', lw=2, label=f'Ensemble (AUC = {roc_auc_ensemble:.2f})')
        
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('(d) ROC Curves')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(True, linestyle='--', alpha=0.7)

        # 3. t-SNE Plot with Density Contours and Support Vectors
        ax_tsne = fig.add_subplot(gs[1, 0:2])
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_features_train = tsne.fit_transform(train_features)
        tsne_features_test = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(test_features)
        
        svm_tsne = SVC(kernel='rbf', class_weight=class_weights, random_state=42)
        svm_tsne.fit(tsne_features_train, y_train)
        
        scatter = ax_tsne.scatter(tsne_features_test[:, 0], tsne_features_test[:, 1], c=y_test, cmap='Set1', alpha=0.7, s=50)
        scatter.set_clim(-0.5, 1.5)
        fig.colorbar(scatter, ax=ax_tsne, ticks=[0, 1], label='Class')

        topo_points = tsne_features_test[y_test == 0]
        triv_points = tsne_features_test[y_test == 1]
        try:
            if len(topo_points) > 5:
                topo_kde = gaussian_kde(topo_points.T)
                x, y = np.meshgrid(np.linspace(topo_points[:, 0].min(), topo_points[:, 0].max(), 100),
                                   np.linspace(topo_points[:, 1].min(), topo_points[:, 1].max(), 100))
                z = topo_kde(np.vstack([x.ravel(), y.ravel()])).reshape(x.shape)
                ax_tsne.contour(x, y, z, levels=3, colors='red', alpha=0.5, linestyles='dashed')
            else:
                logging.warning("Not enough Topological points for density contours")
        except Exception as e:
            logging.warning(f"Failed to compute density contours for Topological: {e}")
        try:
            if len(triv_points) > 5:
                triv_kde = gaussian_kde(triv_points.T)
                x, y = np.meshgrid(np.linspace(triv_points[:, 0].min(), triv_points[:, 0].max(), 100),
                                   np.linspace(triv_points[:, 1].min(), triv_points[:, 1].max(), 100))
                z = triv_kde(np.vstack([x.ravel(), y.ravel()])).reshape(x.shape)
                ax_tsne.contour(x, y, z, levels=3, colors='blue', alpha=0.5, linestyles='dashed')
            else:
                logging.warning("Not enough Trivial points for density contours")
        except Exception as e:
            logging.warning(f"Failed to compute density contours for Trivial: {e}")

        support_indices_test = np.where(np.isin(np.arange(len(y_test)), support_indices))[0]
        topo_support = [idx for idx in support_indices_test if y_test[idx] == 0][:1]
        triv_support = [idx for idx in support_indices_test if y_test[idx] == 1][:1]
        support_examples = topo_support + triv_support
        for idx in support_examples:
            ax_tsne.scatter(tsne_features_test[idx, 0], tsne_features_test[idx, 1], s=150, marker='*', 
                           c='black', label='Support Vector' if idx == support_examples[0] else "")

        x_min, x_max = tsne_features_test[:, 0].min() - 1, tsne_features_test[:, 0].max() + 1
        y_min, y_max = tsne_features_test[:, 1].min() - 1, tsne_features_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = svm_tsne.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax_tsne.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
        ax_tsne.set_title('(e) t-SNE with SVM Decision Boundary')
        ax_tsne.set_xlabel('t-SNE 1')
        ax_tsne.set_ylabel('t-SNE 2')
        ax_tsne.legend(loc='best')
        ax_tsne.grid(True, linestyle='--', alpha=0.7)

        # 4. Grad-CAM Visualizations with Conductance Profiles
        model.eval()
        topo_indices = np.where(y_test == 0)[0]
        triv_indices = np.where(y_test == 1)[0]
        if len(topo_indices) == 0 or len(triv_indices) == 0:
            logging.warning("Not enough samples for Grad-CAM visualization. Skipping Grad-CAM plots.")
            selected_indices = []
        else:
            selected_indices = np.concatenate([topo_indices[:1], triv_indices[:1]])
        
        for i, idx in enumerate(selected_indices):
            ax_gradcam = fig.add_subplot(gs[1, 2 + i])
            img = torch.tensor(X_test[idx:idx+1], dtype=torch.float32)
            logging.info(f"Grad-CAM example {idx}: img shape {img.shape}")
            true_label = y_test[idx]
            pred_label = cnn_pred[idx]
            heatmap = get_gradcam_heatmap(model, img, pred_label, device)
            
            im = ax_gradcam.imshow(img[0, 0].numpy(), cmap='viridis', extent=[-60, 60, 0.28, -0.28], aspect='auto')
            cbar = fig.colorbar(im, ax=ax_gradcam, label='Conductance (2e^2/h)')
            cbar.set_ticks([0, 0.5, 1])
            ax_gradcam.imshow(heatmap, cmap='jet', alpha=0.4, extent=[-60, 60, 0.28, -0.28], aspect='auto')
            ax_gradcam.contour(heatmap, levels=[0.3, 0.6], colors='white', extent=[-60, 60, 0.28, -0.28], alpha=0.8, linestyles='--')
            if true_label == 0:
                ax_gradcam.axvspan(-10, 10, alpha=0.2, color='yellow', label='Expected Edge States')
            ax_gradcam.set_xlabel('Chemical Potential (mu, meV)')
            ax_gradcam.set_ylabel('Voltage Bias (V, meV)')
            ax_gradcam.set_title(f"(f{i+1}) Grad-CAM: True={['Topological', 'Trivial'][true_label]}\nPred={['Topological', 'Trivial'][pred_label]}")
            ax_gradcam.legend(loc='upper right')
            ax_gradcam.grid(True, linestyle='--', alpha=0.7)

            ax_profile = fig.add_subplot(gs[2, 2 + i])
            ax_profile.clear()
            conductance_map = img[0, 0].numpy()
            profile = np.mean(conductance_map, axis=0)
            logging.info(f"Grad-CAM example {idx}: conductance_map shape {conductance_map.shape}, profile shape {profile.shape}")
            mu_values = np.linspace(-60, 60, conductance_map.shape[1])
            ax_profile.plot(mu_values, profile, color='blue', label='Conductance')
            ax_profile.set_xlabel('Chemical Potential (mu, meV)')
            ax_profile.set_ylabel('Avg Conductance (2e^2/h)')
            ax_profile.set_title(f"(g{i+1}) Conductance Profile")
            ax_profile.grid(True, linestyle='--', alpha=0.7)
            if true_label == 0:
                ax_profile.axvspan(-10, 10, alpha=0.2, color='yellow', label='Expected Edge States')
            ax_profile.legend(loc='upper right')

        # 5. Support Vector Examples
        if len(support_examples) < 2:
            logging.warning("Not enough support vector examples to plot. Skipping Support Vector plots.")
        else:
            for i, idx in enumerate(support_examples):
                ax_support = fig.add_subplot(gs[3, i])
                img = torch.tensor(X_test[idx:idx+1], dtype=torch.float32)
                logging.info(f"Support vector example {idx}: img shape {img.shape}")
                true_label = y_test[idx]
                im = ax_support.imshow(img[0, 0].numpy(), cmap='viridis', extent=[-60, 60, 0.28, -0.28], aspect='auto')
                cbar = fig.colorbar(im, ax=ax_support, label='Conductance (2e^2/h)')
                cbar.set_ticks([0, 0.5, 1])
                if true_label == 0:
                    ax_support.axvspan(-10, 10, alpha=0.2, color='yellow', label='Expected Edge States')
                ax_support.set_xlabel('Chemical Potential (mu, meV)')
                ax_support.set_ylabel('Voltage Bias (V, meV)')
                ax_support.set_title(f"(h{i+1}) Support Vector: {['Topological', 'Trivial'][true_label]}")
                ax_support.legend(loc='upper right')
                ax_support.grid(True, linestyle='--', alpha=0.7)

        # 6. Synthetic Phase Diagram
        ax_phase = fig.add_subplot(gs[3, 2:4])
        mu_range = np.linspace(-60, 60, 100)
        v_range = np.linspace(-0.28, 0.28, 100)
        mu_grid, v_grid = np.meshgrid(mu_range, v_range)
        
        synthetic_features = []
        for i, v in enumerate(v_range):
            for j, mu in enumerate(mu_range):
                try:
                    map = np.zeros((30, 30))
                    mu_idx = int((mu + 60) * 30 / 120)
                    v_idx = int((0.28 - v) * 30 / 0.56)
                    if 0 <= mu_idx < 30 and 0 <= v_idx < 30:
                        map[v_idx, mu_idx] = 1.0
                    map = gaussian_filter(map, sigma=2)
                    fft = np.fft.fft2(map)
                    fft_magnitude = np.abs(fft).flatten()
                    fft_magnitude[0] = 0
                    top_fft = fft_magnitude[np.argsort(fft_magnitude)[-10:]]
                    mean = np.mean(map)
                    variance = np.var(map)
                    skewness = np.mean((map - mean) ** 3) / (np.var(map) ** 1.5 + 1e-8)
                    profile = np.mean(map, axis=0)
                    peaks, properties = scipy.signal.find_peaks(profile, height=0.1, prominence=0.05)
                    num_peaks = len(peaks)
                    avg_peak_height = np.mean(properties['peak_heights']) if len(peaks) > 0 else 0.0
                    cnn_features = np.zeros(512)
                    extra_features = np.concatenate([top_fft, [mean, variance, skewness, num_peaks, avg_peak_height]])
                    synthetic_features.append(np.concatenate([cnn_features, extra_features]))
                except Exception as e:
                    logging.error(f"Error in synthetic features computation at v={v}, mu={mu}: {e}")
                    synthetic_features.append(np.zeros(527))  # Fallback to zero vector
        synthetic_features = np.array(synthetic_features)
        logging.info(f"Synthetic features shape: {synthetic_features.shape}")
        
        svm_synth_probs = svm.predict_proba(synthetic_features)
        cnn_synth_probs = np.zeros_like(svm_synth_probs)
        phase_probs = w_cnn * cnn_synth_probs + w_svm * svm_synth_probs
        phase_pred = np.argmax(phase_probs, axis=1).reshape(len(v_range), len(mu_range))
        logging.info(f"Phase pred shape: {phase_pred.shape}")
        im = ax_phase.imshow(phase_pred, cmap='Set1', extent=[-60, 60, -0.28, 0.28], aspect='auto')
        fig.colorbar(im, ax=ax_phase, ticks=[0, 1], label='Predicted Phase')
        ax_phase.set_xlabel('Chemical Potential (mu, meV)')
        ax_phase.set_ylabel('Voltage Bias (V, meV)')
        ax_phase.set_title('(i) Synthetic Phase Diagram (Ensemble)')
        ax_phase.grid(True, linestyle='--', alpha=0.7)

        # 7. Feature Importance Plot
        ax_importance = fig.add_subplot(gs[4, :2])
        top_features_idx = np.argsort(feature_importance)[-5:]
        top_features = [feature_names[i] for i in top_features_idx]
        top_importance = feature_importance[top_features_idx]
        ax_importance.barh(top_features, top_importance, color='skyblue')
        ax_importance.set_xlabel('Permutation Importance')
        ax_importance.set_title('(j) Top 5 Feature Importance (SVM)')
        ax_importance.grid(True, linestyle='--', alpha=0.7)

        # 8. Misclassified Examples with Conductance Profiles
        misclassified_indices = np.where(y_test != ensemble_pred)[0]
        topo_misclassified = [idx for idx in misclassified_indices if y_test[idx] == 0][:1]
        triv_misclassified = [idx for idx in misclassified_indices if y_test[idx] == 1][:1]
        misclassified_examples = topo_misclassified + triv_misclassified
        
        logging.info(f"Number of misclassified examples: {len(misclassified_examples)}")
        if len(misclassified_examples) < 2:
            logging.warning("Not enough misclassified examples to plot. Need at least one topological and one trivial. Skipping Misclassified Examples plots.")
        else:
            for i, idx in enumerate(misclassified_examples):
                ax_mis = fig.add_subplot(gs[4, 2 + i])
                img = torch.tensor(X_test[idx:idx+1], dtype=torch.float32)
                logging.info(f"Misclassified example {idx}: img shape {img.shape}")
                true_label = y_test[idx]
                pred_label = ensemble_pred[idx]
                conductance_map = img[0, 0].numpy()
                logging.info(f"Misclassified example {idx}: conductance_map shape {conductance_map.shape}")
                if conductance_map.ndim != 2:
                    logging.error(f"Conductance map for misclassified example {idx} is not 2D: shape {conductance_map.shape}")
                    continue
                im = ax_mis.imshow(conductance_map, cmap='viridis', extent=[-60, 60, 0.28, -0.28], aspect='auto')
                cbar = fig.colorbar(im, ax=ax_mis, label='Conductance (2e^2/h)')
                cbar.set_ticks([0, 0.5, 1])
                if true_label == 0:
                    ax_mis.axvspan(-10, 10, alpha=0.2, color='yellow', label='Expected Edge States')
                ax_mis.set_xlabel('Chemical Potential (mu, meV)')
                ax_mis.set_ylabel('Voltage Bias (V, meV)')
                ax_mis.set_title(f"(k{i+1}) Misclassified: True={['Topological', 'Trivial'][true_label]}\nPred={['Topological', 'Trivial'][pred_label]}")
                ax_mis.legend(loc='upper right')
                ax_mis.grid(True, linestyle='--', alpha=0.7)

                ax_mis_profile = fig.add_subplot(gs[5, 2 + i])
                ax_mis_profile.clear()
                profile = np.mean(conductance_map, axis=0)
                logging.info(f"Misclassified example {idx}: profile shape {profile.shape}")
                mu_values = np.linspace(-60, 60, conductance_map.shape[1])
                ax_mis_profile.plot(mu_values, profile, color='blue', label='Conductance')
                ax_mis_profile.set_xlabel('Chemical Potential (mu, meV)')
                ax_mis_profile.set_ylabel('Avg Conductance (2e^2/h)')
                ax_mis_profile.set_title(f"(l{i+1}) Misclassified Conductance Profile")
                ax_mis_profile.grid(True, linestyle='--', alpha=0.7)
                if true_label == 0:
                    ax_mis_profile.axvspan(-10, 10, alpha=0.2, color='yellow', label='Expected Edge States')
                ax_mis_profile.legend(loc='upper right')

        plt.savefig(os.path.join(results_dir, "results_summary.png"), bbox_inches='tight', transparent=True)
        plt.close()

        logging.info("Physical Insights from Grad-CAM:")
        for idx in selected_indices:
            true_label = y_test[idx]
            pred_label = cnn_pred[idx]
            img = torch.tensor(X_test[idx:idx+1], dtype=torch.float32)
            heatmap = get_gradcam_heatmap(model, img, pred_label, device)
            high_attention_regions = np.where(heatmap > 0.6)
            if len(high_attention_regions[0]) > 0:
                avg_mu = np.mean(high_attention_regions[1] * 120 / 30 - 60)
                avg_v = np.mean((30 - high_attention_regions[0]) * 0.56 / 30 - 0.28)
                logging.info(f"Sample {idx} (True: {['Topological', 'Trivial'][true_label]}, Pred: {['Topological', 'Trivial'][pred_label]}): "
                             f"High attention at mu ≈ {avg_mu:.2f} meV, V ≈ {avg_v:.2f} meV")
            else:
                logging.info(f"Sample {idx}: No significant attention regions identified.")

        logging.info("Pipeline completed successfully")

    except Exception as e:
        logging.error(f"Error in run_pipeline: {e}")
        raise e

if __name__ == "__main__":
    label_dir = "/home/levi/mzm_project/data"
    image_dir = "/home/levi/mzm_project/images"
    run_pipeline(label_dir, image_dir, num_samples=4000, filename_pattern="conductance_image_{id}.png")