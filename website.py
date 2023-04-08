from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the pre-trained machine learning model
lda_model = pickle.load(open('Main/model.pkl', 'rb'))


# Define the home page route
@app.route("/", methods=['GET', 'POST'])
def searchTopic():
    return render_template("index.html")


# Define the route for processing user input
@app.route("/predict", methods=['GET', 'POST'])
def searchResults():
    # Get the input text from the form
    input_text = request.form.get("input_text")

    # Process the input text with the machine learning model
    label = lda_model.predict([input_text])[0]

    # Render the prediction result on a new page
    return render_template("index.html", label=label)


if __name__ == "__main__":
    app.run(debug=False)
