# Iris Flower Classification Web App (Streamlit)

This project is an interactive web application for classifying Iris flowers using machine learning, built with **Python** and **Streamlit**. The app allows users to explore the Iris dataset, visualize feature relationships, evaluate multiple models, and make real-time predictions — all within an intuitive browser-based interface.

---

## Features

* **Clean, modern UI** powered by Streamlit
* **Automatic dataset loading** from `iris_classification.csv`
* **Interactive input sliders** for entering flower measurements
* **Tabbed navigation** for:

  * Data Overview
  * Visualizations (scatter plots, box plots, species distribution)
  * Model Performance Comparison (KNN, Decision Tree, Naive Bayes)
  * Real-time Prediction Output
* **Dynamic visualizations** using Matplotlib and Seaborn
* **All ML logic handled in Python**

---

## Project Structure

```
iris_classification/
├── iris_streamlit_app.py      # Main Streamlit application
├── iris_classification.csv    # Dataset file (CSV)
└── (optional assets, e.g., plot outputs or saved models)
```

---

## Requirements

* Python 3.7+
* streamlit
* pandas
* scikit-learn
* matplotlib
* seaborn

Install dependencies using:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn
```

---

## How to Run

1. Ensure `iris_classification.csv` is in the same directory as `iris_streamlit_app.py`.
2. Launch the Streamlit app using:

   ```bash
   streamlit run iris_streamlit_app.py
   ```
3. A browser window will open automatically. If not, navigate to:

   ```
   http://localhost:8501
   ```
4. Use the sidebar to input flower measurements and navigate through tabs to explore the app.

---

## Customization

* Add new models or expand to ensemble methods
* Modify charts and themes for improved UX
* Integrate explanations (e.g., SHAP) for model transparency
* Extend to handle image-based flower classification in future iterations

---

## Notes

* Models used: **K-Nearest Neighbors**, **Decision Tree**, **Naive Bayes**
* No JavaScript or frontend libraries are used — everything is built in Python
* The project is structured for **educational and prototyping** purposes
