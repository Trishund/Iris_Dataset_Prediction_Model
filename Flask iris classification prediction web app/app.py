from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load data and train model once
df = pd.read_csv('iris_classification.csv')
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[features]
y = df['species']
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        sl = float(request.form['sepal_length'])
        sw = float(request.form['sepal_width'])
        pl = float(request.form['petal_length'])
        pw = float(request.form['petal_width'])
        pred = knn.predict([[sl, sw, pl, pw]])[0]
        prediction = f'Predicted species: {pred}'

        # Generate and save a plot
        plt.figure(figsize=(6,4))
        sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species')
        plt.scatter([sl], [sw], color='red', s=100, label='Your Input')
        plt.legend()
        plt.title('Sepal Length vs Sepal Width')
        plt.savefig('static/plot.png')
        plt.close()

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)