# Iris Flower Classification Web App (Flask)

This project is a web application for classifying Iris flowers using machine learning, built with **Python (Flask)** for the backend and **HTML (Jinja2 templates)** for the frontend. Users can input flower measurements, visualize the dataset, and receive real-time species predictions using a trained K-Nearest Neighbors (KNN) model.

---

## Features

* **User-friendly interface** for entering flower data
* **Real-time prediction** of Iris species: Setosa, Versicolor, or Virginica
* **Interactive data visualization**: See your input plotted against the dataset
* Fully **Python-powered ML logic** using Scikit-learn and Matplotlib
* Custom dataset support (`iris_classification.csv`)

---

## Project Structure

```
iris_classification/
├── app.py                   # Main Flask application
├── iris_classification.csv  # Dataset file
├── static/
│   └── plot.png             # Generated visual output
└── templates/
    └── index.html           # HTML interface (Jinja2)
```

---

## Requirements

* Python 3.7+
* Flask
* pandas
* scikit-learn
* matplotlib
* seaborn

Install dependencies using:

```bash
pip install flask pandas scikit-learn matplotlib seaborn
```

---

## How to Run the App

1. Place `iris_classification.csv` in the root directory.
2. Ensure the following folders exist:

   * `templates/` with `index.html`
   * `static/` for generated plots (automatically created if not present)
3. Run the application:

   ```bash
   python app.py
   ```
4. Open your browser and navigate to:

   ```
   http://127.0.0.1:5000
   ```
5. Enter values for sepal and petal measurements and click **Predict** to view the predicted species and comparison plot.

---

## Customization

* Update `index.html` to change the web interface design or layout.
* Modify the KNN logic or try alternative ML models in `app.py`.
* Add new visualizations to enrich user experience.

---

## Notes

* This app uses the **K-Nearest Neighbors (KNN)** algorithm.
* All processing, including model training and plotting, happens in Python.
* The goal is to demonstrate ML model deployment using Flask for small-scale apps or prototypes.

