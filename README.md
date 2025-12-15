# Step-by-Step ML Algorithm Visualizer

## Project Overview

The **Step-by-Step ML Algorithm Visualizer** is a web application built using Python Flask and scikit-learn. Its primary goal is to serve as an educational tool for students to understand the mechanics of fundamental machine learning classification and clustering algorithms.

Unlike standard ML libraries that hide the process, this application breaks down complex algorithms (like Gradient Descent in Logistic Regression or centroid shifting in K-Means) into discrete steps, visualizing the current state of the model and the data at each stage using dynamic Matplotlib plots.

## Features

* **Algorithm Selection:** Choose from 5 core ML algorithms:
    * **K-Nearest Neighbors (KNN)**
    * **Decision Tree (DT)**
    * **Random Forest (RF)**
    * **Logistic Regression (LR)**
    * **K-Means Clustering**
* **Dataset Options:** Use classic scikit-learn datasets (Iris, Breast Cancer, Wine Quality).
* **Step-by-Step Visualization:** Observe how the decision boundary or cluster centers change over iterations or splits.
* **Dynamic Parameters:** Adjust key hyperparameters (e.g., K, Max Depth, Learning Rate) before running the visualization.
* **Clean UI:** Professional and intuitive user interface built with Flask and CSS.

## Installation and Setup

### Prerequisites

You need Python 3.8+ installed on your system.

### 1. Clone the Repository (Simulated)



```bash
# Navigate to your project directory
cd ml_explaination/
```

### 2. Set Up the Virtual Environment (Recommended)
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the environment (Linux/macOS)
source venv/bin/activate

# Activate the environment (Windows)
.\venv\Scripts\activate
```

### 3. Install Dependencies
Install all required libraries using the provided requirements.txt
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Start the Flask server
```bash
python app.py
```

## How to use the Application

- Access the Home Page: Open http://127.0.0.1:5000/ in your web browser.

- Select Configuration: Choose an Algorithm and a Dataset.

- Configure Parameters: Input the necessary parameters for the selected algorithm (e.g., set the K value for KNN, or Max Iterations for Logistic Regression).

- Start Visualization: Click the "Start Visualization" button.

- Review Steps: The results page will display the output in a clean, scrollable format, showing:

- A descriptive title for the current stage.

- A brief explanation of the algorithmic action performed.

- A plot demonstrating the result of that action (e.g., a shifting decision boundary or highlighted neighbors).

## Algorithm Implementation Details
To achieve the step-by-step visualization, standard scikit-learn algorithms were sometimes bypassed or manually re-implemented to expose intermediate states
### Logistic Regression
Manual Gradient Descent loop implementation to show the decision boundary shifting after key iterations (max_iter)

### K-Means
Manual E-step (assignment) and M-step (centroid update) loops, showing clusters and centroids after each iteration.

### Decision Tree
Showing the final decision regions plotted at increasing values of max_depth (e.g., depth 1, 2, 3) to illustrate recursive partitioning.

### Random Forest
Showing the individual, high-variance boundaries of the first few Decision Trees, followed by the final, smooth boundary of the aggregated ensemble.

### KNN
Focusing on the prediction phase for a hypothetical test point: calculating distance, highlighting K-neighbors, and showing the final classification vote.

# Contact
## Chandankumar Johakhim Patel
pate.383@wright.edu