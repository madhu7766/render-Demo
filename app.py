from flask import Flask, render_template, request
import pandas as pd
import joblib



app = Flask(__name__)

# Load bundled model
bundle = joblib.load("income_model.pkl")

model = bundle["model"]
scaler = bundle["scaler"]
label_encoders = bundle["encoders"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    data = {
        'age': int(request.form['age']),
        'workclass': request.form['workclass'],
        'fnlwgt': int(request.form['fnlwgt']),
        'education': request.form['education'],
        'educational-num': int(request.form['educational_num']),
        'marital-status': request.form['marital_status'],
        'occupation': request.form['occupation'],
        'relationship': request.form['relationship'],
        'race': request.form['race'],
        'gender': request.form['gender'],
        'capital-gain': int(request.form['capital_gain']),
        'capital-loss': int(request.form['capital_loss']),
        'hours-per-week': int(request.form['hours_per_week']),
        'native-country': request.form['native_country']
    }

    sample = pd.DataFrame([data])

    for col in label_encoders:
        sample[col] = label_encoders[col].transform(sample[col].astype(str))

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    result = ">50K Income" if prediction == 1 else "<=50K Income"
    confidence = round(max(probability) * 100, 2)

    return render_template(
        "index.html",
        prediction_text=f"Prediction: {result} (Confidence: {confidence}%)"
    )

if __name__ == "__main__":
    app.run(debug=True)