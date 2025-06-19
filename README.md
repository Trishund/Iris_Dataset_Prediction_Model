# Iris Flower Classification – Machine Learning Project

This repository contains multiple implementations of a machine learning model developed to classify Iris flower species based on their physical measurements. It is part of a series of projects completed during a Data Science internship at CodSoft.

## Project Overview

The goal of this project is to build a supervised machine learning model that can accurately classify Iris flowers into one of three species:

* Setosa
* Versicolor
* Virginica

These species are distinguishable by measurements of their **sepal length**, **sepal width**, **petal length**, and **petal width**. This classification task serves as a foundational introduction to supervised learning and is a widely used benchmark in the field of machine learning.

To strengthen my learning and enhance my understanding of real-world applications, I implemented this model using multiple technologies and interfaces:

1. **Jupyter Notebook (Core ML development)**
2. **Flask Web Application (Python backend with RESTful interaction)**
3. **Streamlit Application (Rapid prototyping and interactive visualization)**
4. **HTML + JavaScript Web App (Client-side model implementation)**

Each version serves as an independent deployment approach and reflects my intent to explore the different ways machine learning can be integrated into user-facing applications.

## Dataset

* **Source**: [UCI Machine Learning Repository – Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
* **Features**:

  * Sepal length (cm)
  * Sepal width (cm)
  * Petal length (cm)
  * Petal width (cm)
* **Target**: Species (`setosa`, `versicolor`, `virginica`)

## Project Structure

```
.
├── Iris classification model/                      # Jupyter Notebook model
├── Flask iris classification prediction web app/   # Flask-based deployment
├── Streamlit Iris dataset classification .../      # Streamlit UI
├── Web app/                                        # HTML + JavaScript implementation
├── .gitignore
└── README.md
```

## Workflow

### Data Preprocessing

* Dataset loaded into Pandas DataFrame
* Checked for null or duplicate entries
* Features and labels separated for training

### Model Building

* Algorithms tested:

  * K-Nearest Neighbors
  * Decision Tree
  * Gaussian Naive Bayes
* Model evaluation using accuracy scores and confusion matrix

### Deployment

* Built and deployed applications using different tools:

  * **Flask**: Backend API in Python
  * **Streamlit**: Interactive app with real-time predictions
  * **HTML + JavaScript**: Model implemented for browser-based predictions
  * **Jupyter Notebook**: Full exploratory development and analysis

## Learning and Motivation

The foundational implementation was entirely coded by me in **Jupyter Notebook**, where I handled the data analysis, feature selection, model training, and evaluation.

Once I had the core logic in place, I expanded the project by experimenting with deployment and interactivity. I explored frameworks like **Flask** and **Streamlit**, and even attempted client-side machine learning using **HTML + JavaScript**. This hands-on experimentation allowed me to better understand the end-to-end journey of an ML project—from raw data to a deployable application.

Throughout this process, I also leveraged **AI-based coding tools** for suggestions, automation, and error resolution. These tools significantly enhanced the learning curve and helped me explore concepts more efficiently. My goal with this project was to **learn by doing**, make mistakes, iterate fast, and understand the practical nuances of deploying machine learning solutions.

## Technologies Used

* Python 3.x
* Jupyter Notebook
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Flask
* Streamlit
* HTML, CSS, JavaScript

## Acknowledgments

This is Task 3 of the Data Science Internship at CodSoft.
Development was supported by self-study, community tutorials, and AI-assisted development platforms.
