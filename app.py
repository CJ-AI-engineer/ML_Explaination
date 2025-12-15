from flask import Flask, render_template, request, redirect, url_for
from ml_logic.visualizer import MLVisualizer

app = Flask(__name__)

ALGORITHMS = {
    'knn': 'K-Nearest Neighbors (KNN)',
    'dt': 'Decision Tree',
    'rf': 'Random Forest',
    'lr': 'Logistic Regression',
    'kmeans': 'K-Means Clustering'
}

DATASETS = {
    'iris': 'Iris Classification (3 classes)',
    'cancer': 'Breast Cancer Classification (2 classes)',
    'wine': 'Wine Quality (Multi-class/Regression)'
}

@app.route('/', methods=['GET'])
def index():
    """Show the main selection form to the user"""
    return render_template('index.html', 
                           algorithms=ALGORITHMS, 
                           datasets=DATASETS)


@app.route('/results', methods=['POST'])
def results():
    """
    Calculation and displays results.
    """
    algo_key = request.form.get('algorithm')
    dataset_key = request.form.get('dataset')
    
    if not algo_key or not dataset_key:
        return redirect(url_for('index'))

    params = {}
    
    try:
        if algo_key in ['lr', 'kmeans']:
            # for logistic regresssion
            if algo_key == 'lr':
                max_iter = request.form.get('lr_max_iter', type=int)
                lr = request.form.get('lr_rate', type=float)
                if max_iter: params['max_iter'] = max_iter
                if lr: params['lr'] = lr
            
            # for K-Means Algorithm
            if algo_key == 'kmeans':
                n_clusters = request.form.get('kmeans_clusters', type=int)
                if n_clusters: params['n_clusters'] = n_clusters
            
            
        elif algo_key in ['dt', 'rf']:
            max_depth = request.form.get('tree_depth', type=int)
            if max_depth: params['max_depth'] = max_depth

            if algo_key == 'rf':
                n_estimators = request.form.get('rf_estimators', type=int)
                if n_estimators: params['n_estimators'] = n_estimators

        elif algo_key == 'knn':
            n_neighbors = request.form.get('knn_neighbors', type=int)
            if n_neighbors: params['n_neighbors'] = n_neighbors
            
    except ValueError:
        return "Error: Invalid parameter value provided. Please ensure numbers are used.", 400

    try:
        visualizer = MLVisualizer(dataset_key, algo_key, params)
        steps_data = visualizer.get_steps()

        return render_template('results.html', 
                               algorithm=ALGORITHMS[algo_key], 
                               dataset=DATASETS[dataset_key],
                               steps=steps_data)
                               
    except Exception as e:
        app.logger.error(f"ML Visualization Error: {e}")
        return f"<h1>An error occurred during ML processing.</h1><p>Algorithm: {ALGORITHMS[algo_key]}, Dataset: {DATASETS[dataset_key]}</p><p>Details: {e}</p><a href='/'>Go Back</a>", 500
        


if __name__ == '__main__':
    app.run(debug=True)

