from flask import Flask, render_template, request
import numpy as np
import pickle
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction", methods=["GET","POST"])
def predict():
    pred_output = ""
    if request.method == "POST":
        loaded_model = pickle.load(open("model.pkl","rb"))
        height = request.form["height"]
        height = np.array([height], dtype="float64")
        predicted_weight = loaded_model.predict(np.expand_dims(height, axis=1))[0]
        pred_output = f"The predicted weight is {predicted_weight:.2f} pounds"
    return render_template("prediction.html", pred=pred_output)

if (__name__ == "__main__"):
    app.run()
