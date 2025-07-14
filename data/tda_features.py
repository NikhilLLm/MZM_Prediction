import numpy as np
import gudhi as gd
from gudhi.representations import PersistenceImage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_f1_support
import matplotlib.pyplot as plt
import logging
import os
# Set up logging
logging.basicConfig(filename='/home/levi/mzm_project/results/train.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def compute_tda_features(conductance_maps, y, results_dir, samples_to_visualize=None):
    try:
        # Convert conductance maps to point clouds: (i, j, conductance[i, j])
        tda_features = []
        persistence_diagrams = []
        for idx, conductance_map in enumerate(conductance_maps):
            points = []
            for i in range(30):
                for j in range(30):
                    points.append([i, j, conductance_map[i, j]])
            points = np.array(points)

            # Build a Vietoris-Rips complex
            rips_complex = gd.RipsComplex(points=points, max_edge_length=5.0)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
            diag = simplex_tree.persistence()
            persistence_diagrams.append(diag)

            # Convert persistence diagram to persistence image
            pi = PersistenceImage(resolution=[20, 20], im_range=[0, 5, 0, 5])
            pi_feature = pi.fit_transform([diag])[0].flatten()
            tda_features.append(pi_feature)

        tda_features = np.array(tda_features)
        logging.info(f"Computed TDA features for {len(conductance_maps)} samples")

        # Train a Random Forest classifier on TDA features
        X_train, X_test, y_train, y_test = tda_features[:len(y)//2], tda_features[len(y)//2:], y[:len(y)//2], y[len(y)//2:]
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, rf_pred)
        precision, recall, f1, _ = precision_recall_f1_support(y_test, rf_pred, average=None)
        logging.info(f"TDA + Random Forest - Test Accuracy: {accuracy:.3f}")

        # Save TDA metrics
        with open(os.path.join(results_dir, "tda_metrics.txt"), "w") as f:
            f.write("TDA + Random Forest Metrics (Test Set):\n")
            f.write("Model\tAccuracy\tPrecision (Topo)\tRecall (Topo)\tF1 (Topo)\tPrecision (Triv)\tRecall (Triv)\tF1 (Triv)\n")
            f.write(f"TDA+RF\t{accuracy:.3f}\t\t{precision[0]:.3f}\t\t{recall[0]:.3f}\t\t{f1[0]:.3f}\t\t{precision[1]:.3f}\t\t{recall[1]:.3f}\t\t{f1[1]:.3f}\n")

        # Visualize persistence diagrams for selected samples
        if samples_to_visualize is not None:
            os.makedirs(os.path.join(results_dir, "persistence_diagrams"), exist_ok=True)
            for idx in samples_to_visualize:
                diag = persistence_diagrams[idx]
                gd.plot_persistence_diagram(diag)
                plt.title(f"Persistence Diagram - Sample {idx} (Class: {['Topological', 'Trivial'][y[idx]]})")
                plt.savefig(os.path.join(results_dir, f"persistence_diagrams/sample_{idx}.png"))
                plt.close()

        return tda_features, accuracy, precision, recall, f1, rf_pred, y_test
    except Exception as e:
        logging.error(f"Error in compute_tda_features: {e}")
        return None, None, None, None, None, None, None