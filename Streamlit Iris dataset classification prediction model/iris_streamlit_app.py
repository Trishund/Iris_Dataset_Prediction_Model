import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# --- Custom CSS for gradient background and card effect ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%) !important;
    }
    .main > div {
        background: rgba(255,255,255,0.97) !important;
        border-radius: 18px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.2);
        padding: 32px 28px 28px 28px;
        margin-top: 40px;
        max-width: 1100px;
        margin-left: auto;
        margin-right: auto;
    }
    .stTabs [data-baseweb="tab-list"] { justify-content: center; }
    .stTabs [data-baseweb="tab"] { font-size: 1.1em; }
    .stButton>button {
        background: linear-gradient(90deg, #8ec5fc 0%, #e0c3fc 100%);
        color: #222;
        font-weight: 600;
        border-radius: 6px;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #e0c3fc 0%, #8ec5fc 100%);
    }
    </style>
""", unsafe_allow_html=True)

st.title("Iris Flower Classification")

uploaded_file = st.sidebar.file_uploader("Upload your Iris data file (CSV or Excel)", type=["xlsx", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
species = df['species'].unique()

    # --- Sidebar Input ---
    st.sidebar.title("Classify a New Iris Flower")
    with st.sidebar.form("predict_form"):
        sl = st.number_input('Sepal Length (cm)', float(df.sepal_length.min()), float(df.sepal_length.max()), float(df.sepal_length.mean()))
        sw = st.number_input('Sepal Width (cm)', float(df.sepal_width.min()), float(df.sepal_width.max()), float(df.sepal_width.mean()))
        pl = st.number_input('Petal Length (cm)', float(df.petal_length.min()), float(df.petal_length.max()), float(df.petal_length.mean()))
        pw = st.number_input('Petal Width (cm)', float(df.petal_width.min()), float(df.petal_width.max()), float(df.petal_width.mean()))
        submitted = st.form_submit_button("Predict Species")
    user_sample = np.array([[sl, sw, pl, pw]])

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Visualizations", "Models", "Predictions"])

with tab1:
        st.subheader("About the Iris Dataset")
        st.write(
            "The Iris dataset is a classic dataset in machine learning, containing 150 samples of iris flowers from three species: "
            "**setosa**, **versicolor**, and **virginica**. Each sample has four features: sepal length, sepal width, petal length, and petal width (all in cm)."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.markdown("#### Species Distribution")
            st.write(df['species'].value_counts())
            fig, ax = plt.subplots()
            df['species'].value_counts().plot.pie(autopct='%1.0f%%', colors=['#8ec5fc', '#e0c3fc', '#f9a8d4'], ax=ax)
            ax.set_ylabel('')
            st.pyplot(fig)

with tab2:
        st.subheader("Data Visualizations")
    col1, col2 = st.columns(2)
    with col1:
            st.markdown("**Sepal Length vs Sepal Width**")
        fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species', palette='pastel', ax=ax)
        st.pyplot(fig)
    with col2:
            st.markdown("**Petal Length vs Petal Width**")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', palette='pastel', ax=ax)
            st.pyplot(fig)
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Box Plot: Sepal Length**")
        fig, ax = plt.subplots()
            sns.boxplot(data=df, x='species', y='sepal_length', palette='pastel', ax=ax)
        st.pyplot(fig)
        with col4:
            st.markdown("**Box Plot: Petal Width**")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='species', y='petal_width', palette='pastel', ax=ax)
    st.pyplot(fig)

with tab3:
        st.subheader("Machine Learning Models")
    X = df[features]
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_acc = accuracy_score(y_test, knn.predict(X_test))

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test))

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_acc = accuracy_score(y_test, nb.predict(X_test))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### K-Nearest Neighbors (KNN)")
            st.info("KNN classifies a sample based on the majority label among its k closest neighbors.")
            st.write(f"**Accuracy:** {knn_acc:.1%}")
        with col2:
            st.markdown("##### Decision Tree")
            st.info("Decision Trees split the data based on feature values to create a tree structure for classification.")
            st.write(f"**Accuracy:** {dt_acc:.1%}")
        with col3:
            st.markdown("##### Naive Bayes")
            st.info("Naive Bayes uses probability theory and assumes features are independent.")
            st.write(f"**Accuracy:** {nb_acc:.1%}")

with tab4:
        st.subheader("Classify a New Iris Flower")
        if submitted:
    knn_pred = knn.predict(user_sample)[0]
    dt_pred = dt.predict(user_sample)[0]
    nb_pred = nb.predict(user_sample)[0]
            st.markdown("#### Model Predictions")
    st.write(f"**KNN Prediction:** {knn_pred}")
    st.write(f"**Decision Tree Prediction:** {dt_pred}")
    st.write(f"**Naive Bayes Prediction:** {nb_pred}")
    preds = [knn_pred, dt_pred, nb_pred]
    if preds.count(preds[0]) == 3:
        st.success(f"All models agree: **{preds[0]}**")
    else:
        st.warning(f"Models disagree: {', '.join(preds)}")
        else:
            st.info("Enter values in the sidebar and click 'Predict Species' to see predictions.")

else:
    st.sidebar.title("Classify a New Iris Flower")
    st.sidebar.info("Please upload your Iris data file (CSV or Excel) in the sidebar to continue.")
    st.warning("Upload your Iris dataset to get started!")
