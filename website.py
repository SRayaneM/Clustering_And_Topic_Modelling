from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load("Main\Clustering_Using_TF-IDF.ipynb")


# Define the home page route
@app.route("/")
def home():
    return render_template("index.html")


# Define the route for processing user input
@app.route("/predict", methods=["POST"])
def predict():
    # Get the input text from the form
    input_text = request.form.get("input_text")

    # Process the input text with the machine learning model
    label = model.predict([input_text])[0]

    # Render the prediction result on a new page
    return render_template("result.html", label=label)


if __name__ == "__main__":
    app.run(debug=True)
