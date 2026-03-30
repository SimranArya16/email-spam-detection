# app.py
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Set threshold for spam detection
THRESHOLD = 0.35

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email_text = request.form["email_text"]

        # Transform email text using the same vectorizer
        email_vector = vectorizer.transform([email_text])

        # Get probability for Spam class
        spam_prob = model.predict_proba(email_vector)[0][1]
        spam_prob_percent = round(spam_prob * 100, 2)

        # Decide label based on threshold
        if spam_prob > THRESHOLD:
            label = "Spam"
        else:
            label = "Not Spam"

        return render_template(
            "index.html",
            prediction=label,
            probability=spam_prob_percent,
            email=email_text
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)