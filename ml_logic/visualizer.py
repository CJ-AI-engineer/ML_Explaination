import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier 
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import euclidean_distances # For manual distance calc
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 

import pandas as pd
import numpy as np


class MLVisualizer:
    def __init__(self, dataset_name, algorithm_name, params={}):
        self.dataset_name = dataset_name
        self.algorithm_name = algorithm_name
        self.params = params
        self.X, self.y, self.target_names = self._load_data()
        self.steps = []
        self._run_algorithm()

    def _load_data(self):
        """Loads and preprocesses the selected dataset."""
        if self.dataset_name == 'iris':
            data = load_iris()
        elif self.dataset_name == 'cancer':
            data = load_breast_cancer()
        elif self.dataset_name == 'wine':
            # Use sklearn's wine dataset as for defalut
            from sklearn.datasets import load_wine
            data = load_wine()
            
        X = data.data
        y = data.target
        target_names = data.target_names

        # Apply PCA for 2D visualization if features > 2
        if X.shape[1] > 2:
            X = StandardScaler().fit_transform(X)
            X = PCA(n_components=2).fit_transform(X)
        
        return X, y, target_names

    def _run_algorithm(self):
        """Dispatches to the specific algorithm visualization method."""
        if self.algorithm_name == 'kmeans':
            self._visualize_kmeans()
        elif self.algorithm_name == 'dt':
            self._visualize_decision_tree()
        elif self.algorithm_name == 'lr':
            self._visualize_logistic_regression()
        elif self.algorithm_name == 'rf':
            self._visualize_random_forest()
        elif self.algorithm_name == 'knn':
            self._visualize_knn()
        
    def get_steps(self):
        """Returns the list of steps data for the frontend."""
        return self.steps

    
    def _visualize_kmeans(self):
        """K-Means step-by-step Implementation."""
        
        k = self.params.get('n_clusters', 3)
        np.random.seed(42) 
        
        # Step 1: Initialization
        # Randomly select 'k' points as initial centroids
        indices = np.random.choice(len(self.X), k, replace=False)
        centroids = self.X[indices]
        
        self._save_step("Initialization", 
                        "Randomly selected initial centroids.", 
                        self._plot_clusters(None, centroids))
        
        # Step 2: Iterative Refinement
        max_iterations = self.params.get('max_iter', 5)
        for i in range(max_iterations):
            # Assign Points to Nearest Centroid
            distances = np.sqrt(((self.X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            self._save_step(f"Iteration {i+1}: Assignment (E-Step)", 
                            "Each data point is assigned to the nearest centroid.", 
                            self._plot_clusters(labels, centroids))

            # Update Centroids
            new_centroids = np.array([self.X[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else centroids[j]
                                      for j in range(k)])
            
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
            
            self._save_step(f"Iteration {i+1}: Update (M-Step)", 
                            "Centroids are moved to the mean of their assigned cluster points.", 
                            self._plot_clusters(labels, centroids))
            
        self._save_step("Final Result", 
                        f"K-Means converged after {i+1} iterations.", 
                        self._plot_clusters(labels, centroids))
        
    

    def _plot_clusters(self, labels, centroids):
        """Generates a base64 encoded image of the current cluster state."""
        plt.figure(figsize=(8, 6))
        
        if labels is not None:
            scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        else:
            plt.scatter(self.X[:, 0], self.X[:, 1], c='gray', s=50, alpha=0.7)
            
        # Plot centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red', edgecolors='black', label='Centroids')
        
        plt.title(f"{self.dataset_name.title()} Data (PCA reduced)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"data:image/png;base64,{data}"

    def _save_step(self, title, description, plot_data):
        self.steps.append({
            'title': title,
            'description': description,
            'image_data': plot_data
        })
        
    def _visualize_decision_tree(self):
        """Visualizes the Decision Tree splitting process by showing decision regions at increasing depth."""

        user_max_depth = self.params.get('max_depth', 5)

        stages = [1]
        if user_max_depth >= 2: stages.append(2)
        if user_max_depth >= 3: stages.append(3)
        if user_max_depth >= 5: stages.append(5)
        
        if user_max_depth not in stages:
            stages.append(user_max_depth)
        stages = sorted(list(set(stages))) # Remove duplicates and sort

        X, y = self.X, self.y.astype(int)

        self._save_step("Step 1: Initialization", 
                        "The tree starts with a single, un-split node, representing the entire dataset. No classification is performed yet.",
                        self._plot_decision_regions(None, X, y, 'Initial State (Depth 0)'))

        for depth in stages:
            model = DecisionTreeClassifier(max_depth=depth, random_state=42)
            model.fit(X, y)

            description = (
                f"The tree splits recursively until maximum depth {depth} is reached. "
                "Each split is chosen to maximize the purity of the resulting regions (Information Gain/Gini)."
            )
            self._save_step(f"Step {depth+1}: Max Depth = {depth}", 
                            description,
                            self._plot_decision_regions(model, X, y, f'Decision Regions (Max Depth {depth})'))
                        
        
    def _plot_decision_regions(self, classifier, X, y, title):
        plt.figure(figsize=(8, 6))
        
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFAAFF']) # I am considering only up to 4 classes
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FF00FF'])
        
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        
        if classifier:
            Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto', alpha=0.6)
        else:
            plt.pcolormesh(xx, yy, np.zeros_like(xx), cmap=cmap_light, shading='auto', alpha=0.1)

        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=40)
        
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(title)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True, linestyle=':', alpha=0.6)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"data:image/png;base64,{data}"
        
    def _plot_decision_tree_structure(self, model):
        """Creates the image for the final Decision Tree structure."""
        return "placeholder_tree_image_data"
    
    def _visualize_logistic_regression(self):
        """Implements Logistic Regression step-by-step using a simplified Gradient Descent."""
       
        X = self.X
        y = self.y.astype(int) 
        
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Initialize weights (weight = theta) randomly
        np.random.seed(42)
        theta = np.random.rand(X.shape[1]) * 0.1
        
        # Hyperparameters
        learning_rate = self.params.get('lr', 0.01)
        max_iterations = self.params.get('max_iter', 200) 
        # First Initial State
        self._save_step("Step 1: Initialization",
                        "Weights (coefficients) are initialized randomly. The decision boundary is arbitrary.",
                        self._plot_decision_boundary(X, y, theta, 'Initial State'))

        # Gradient Descent Loop
        report_steps = [1, 5, 20, 50, 100, max_iterations]
        
        for i in range(1, max_iterations + 1):
            
            # 1. Calculate Hypothesis (h(x) = Sigmoid(X * theta))
            z = X.dot(theta)
            h = 1 / (1 + np.exp(-z)) # The Sigmoid Function
            
            # 2. Calculate Error and Gradient
            error = h - y
            gradient = X.T.dot(error) / len(y)
            
            # 3. Update Weights (theta)
            theta -= learning_rate * gradient
            
            if i in report_steps:
                description = (
                    f"Iteration {i}: The model calculated the error (h(x) - y) and updated the weights "
                    f"using Gradient Descent. The boundary is shifting towards optimal separation."
                )
                self._save_step(f"Iteration {i}: Updating Boundary",
                                description,
                                self._plot_decision_boundary(X, y, theta, f'Iteration {i}'))

        self._save_step("Final Result", 
                        f"Gradient Descent converged after {max_iterations} iterations, finding the optimal decision boundary.", 
                        self._plot_decision_boundary(X, y, theta, 'Final Model'))


    def _plot_decision_boundary(self, X, y, theta, title):
        """Generates a base64 encoded image of the current decision boundary."""
        
        X_plot = X[:, 1:] 
        
        plt.figure(figsize=(8, 6))
        
        # Plot the Data Points and provide colored of true class
        scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
        
 
        x1_min, x1_max = X_plot[:, 0].min() - 0.1, X_plot[:, 0].max() + 0.1
        plot_x = np.array([x1_min, x1_max])
        
        # Check if theta_2 is close to zero to avoid division by zero
        #TODO: I need to think on getting it dynamically
        if np.abs(theta[2]) > 1e-9:
            plot_y = (-theta[0] - theta[1] * plot_x) / theta[2]
            plt.plot(plot_x, plot_y, c='red', linestyle='--', label='Decision Boundary')
        else:
            plt.axvline(x=-theta[0]/theta[1], c='red', linestyle='--', label='Decision Boundary (Vertical)')
            
        plt.title(f"Logistic Regression Decision Boundary ({title})")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"data:image/png;base64,{data}"
    
    def _visualize_random_forest(self):
        """Visualizes the Random Forest """

        X, y = self.X, self.y.astype(int)
        
        n_estimators = self.params.get('n_estimators', 5) 
        max_depth = self.params.get('max_depth', 3)       # Depth for the base trees is 3 if not given

        self._save_step("Step 1: Ensemble Initialization", 
                        f"The Random Forest is a collection of Decision Trees. We will train {n_estimators} trees, each using a random subset of data (Bagging) and features.",
                        self._plot_decision_regions(None, X, y, 'Initial State (No Trees Trained)'))

    
        full_model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42,
        )
        full_model.fit(X, y)
        
        trees_to_show = min(3, n_estimators)
        
        for i in range(trees_to_show):
            individual_tree = full_model.estimators_[i]
            
            description = (
                f"Tree {i+1} is trained on a bootstrapped sample of the data and considers only a random subset "
                "of features at each split. Notice how its boundary is distinct from others."
            )
            self._save_step(f"Step {i+2}: Individual Decision Tree {i+1}", 
                            description,
                            self._plot_decision_regions(individual_tree, X, y, f'Decision Tree {i+1} Boundary'))
            
         
        description = (
            f"The final prediction is made by averaging the predictions of all {n_estimators} individual trees "
            "('Voting'). This process cancels out the high variance (overfitting) of individual trees, resulting "
            "in a smoother, more robust decision boundary."
        )
        self._save_step(f"Step {trees_to_show + 2}: Ensemble Aggregation (Final)", 
                        description,
                        self._plot_decision_regions(full_model, X, y, f'Random Forest Final Boundary'))
        
        
    def _visualize_knn(self):
        """Visualizes the KNN prediction process for a single test point."""

        X, y = self.X, self.y.astype(int)
        
        n_val = self.params.get('n_neighbors', 5) 
        test_point = X.mean(axis=0).reshape(1, -1)
        
        knn_model = KNeighborsClassifier(n_neighbors=n_val) 
        knn_model.fit(X, y)

        self._save_step("Step 1: Training (Lazy)", 
                        "KNN training is trivial...",
                        self._plot_knn_state(X, y, test_point, None, None, 'Training Data Stored'))
        
        
        # Calculate Euclidean distances from the test point to all training points
        distances = euclidean_distances(X, test_point).flatten()
        
        self._save_step("Step 2: Distance Calculation", 
                        "The distance (e.g., Euclidean) from the new test point (red star) to every existing data point is calculated.",
                        self._plot_knn_state(X, y, test_point, distances, None, 'Distances Calculated'))
        
        
        k_indices = np.argsort(distances)[:n_val]
        
        self._save_step(f"Step 3: Selecting K={n_val} Neighbors", 
                        f"The {n_val} points with the shortest distances are identified as the nearest neighbors (highlighted with circles).",
                        self._plot_knn_state(X, y, test_point, distances, k_indices, 'K-Nearest Neighbors Selected'))

        # Get the labels of the k neighbors
        neighbor_labels = y[k_indices]
        # Find the most frequent class among the neighbors (the final prediction)
        (values, counts) = np.unique(neighbor_labels, return_counts=True)
        prediction_index = np.argmax(counts)
        prediction = values[prediction_index]
        
        predicted_class_name = self.target_names[prediction]
        
        description = (
            f"The majority class among the {n_val} neighbors dictates the final prediction. " 
            f"Prediction: **{predicted_class_name}** (Vote Count: {counts[prediction_index]})"
        )
        self._save_step("Step 4: Final Voting and Prediction", 
                        description,
                        self._plot_knn_state(X, y, test_point, distances, k_indices, f'Predicted Class: {predicted_class_name}', prediction_class=prediction, n_val=n_val))

    
    def _plot_knn_state(self, X, y, test_point, distances, k_indices, title, prediction_class=None, n_val=5):
        """Generates a base64 encoded image of the current KNN stage"""
        
        plt.figure(figsize=(8, 6))
        
        cmap_bold = ListedColormap(['#FF0000', '#0000FF', '#00FF00', '#FF00FF']) 
        
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=50, alpha=0.7, label='Training Data')
        
        if prediction_class is not None:
             # Plot the test point colored by the final prediction
            predicted_color = cmap_bold(prediction_class / (len(self.target_names) - 1))
            plt.scatter(test_point[:, 0], test_point[:, 1], marker='*', s=300, c=[predicted_color], edgecolors='black', linewidth=1.5, label='Test Point (Predicted)')
        else:
            # Plot the test point uisng black or gray
            plt.scatter(test_point[:, 0], test_point[:, 1], marker='*', s=300, c='red', edgecolors='black', linewidth=1.5, label='Test Point')
        
 
        if k_indices is not None:
            # Draw a circle around the k-nearest points
            knn_points = X[k_indices]
            plt.scatter(knn_points[:, 0], knn_points[:, 1], marker='o', facecolors='none', edgecolors='gold', s=200, linewidth=2, label='K Neighbors')
            
            for idx in k_indices:
                plt.plot([test_point[0, 0], X[idx, 0]], [test_point[0, 1], X[idx, 1]], 'k--', alpha=0.3)
        
        plt.title(f"KNN Classification (K={n_val}): {title}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.6)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"data:image/png;base64,{data}"