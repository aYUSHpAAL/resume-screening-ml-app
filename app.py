from flask import Flask, render_template, request
import pickle
import PyPDF2
import numpy as np

app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = pickle.load(open("model/resume_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    top_predictions = None

    if request.method == "POST":
        pdf = request.files["resume_file"]
        text = extract_text_from_pdf(pdf)

        vec = vectorizer.transform([text])

        probabilities = model.predict_proba(vec)[0]
        classes = model.classes_

        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {
                "category": classes[i],
                "probability": round(
                    (probabilities[i] / probabilities.sum()) * 100, 2
                )
            }
            for i in top_indices
        ]

        prediction = top_predictions[0]["category"]
        confidence = top_predictions[0]["probability"]

    return render_template(
        "index.html",
        result=prediction,
        confidence=confidence,
        top_predictions=top_predictions
    )

if __name__ == "__main__":
    app.run(debug=True)
