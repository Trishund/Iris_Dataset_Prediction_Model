# Iris Flower Dataset Analysis

This project is a complete data science workflow applied to the classic **Iris Flower Dataset**, including **data visualization**, **exploratory data analysis (EDA)**, **model training**, **evaluation**, and **comparison of multiple classification algorithms**. It is built using Python in a Jupyter Notebook environment.

---

## Project Objectives

* Understand and visualize the structure of the Iris dataset.
* Preprocess and explore the data to extract meaningful patterns.
* Train multiple supervised machine learning models to classify Iris species.
* Evaluate model performance and compare accuracy.
* Identify key features contributing to classification using feature importance.

---

## Dataset

* **Source**: [Kaggle – Iris Flower Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)
* **Features**:

  * `sepal_length`
  * `sepal_width`
  * `petal_length`
  * `petal_width`
* **Target**:

  * `species` (Setosa, Versicolor, Virginica)

---

## Exploratory Data Analysis (EDA)

* Dataset overview: shape, columns, types, statistical summary
* Class balance analysis (`species` distribution)
* Histograms for feature distributions grouped by species
* Pairplots to examine relationships between variables
* Box plots to compare feature values by species
* Correlation heatmap to understand feature interdependencies

---

## Machine Learning Models

The following supervised classifiers were trained and evaluated:

* **Random Forest Classifier**
* **Logistic Regression**
* **Support Vector Machine (SVM)**

Each model was:

* Trained on a stratified 80/20 train-test split
* Evaluated using accuracy, classification report, and confusion matrix
* Visualized with confusion matrix heatmaps

---

## Model Performance & Comparison

* Accuracy scores are compared across all models
* A bar chart visualizes model accuracy side-by-side
* All models achieved **>90% accuracy**, with Random Forest performing slightly better

---

## Feature Importance (Random Forest)

* Calculated and visualized feature importance from the trained Random Forest model
* Identified **petal length** and **petal width** as the most significant features for classification

---

## Technologies Used

* **Python 3.x**
* **Pandas** & **NumPy** for data manipulation
* **Matplotlib** & **Seaborn** for data visualization
* **Scikit-learn** for machine learning
* **Jupyter Notebook** for interactive coding

---

## Key Findings

1. The dataset is clean with no missing or corrupted values.
2. **Petal length** and **petal width** are the most important features for classification.
3. All trained models achieved high classification accuracy (>90%).
4. Setosa, Versicolor, and Virginica are well-separated in feature space.

---

## Learning Goals

This project was developed to:

* Practice complete machine learning workflow from scratch
* Strengthen understanding of EDA and model evaluation
* Gain hands-on experience with classification tasks
* Experiment with different models and feature importance analysis

---

## File Overview

```
Iris_Flower_Dataset_Analysis/
├── IRIS.csv                          # Dataset file
├── Iris_Analysis_Notebook.ipynb      # Jupyter Notebook (main analysis)
└── README.md                         # This file

