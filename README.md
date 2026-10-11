# Churn Intelligence

An interactive customer churn prediction web app. Enter a customer profile and get the churn probability, a detailed explanation of the drivers, and a tailored loyalty and retention plan.

**Live demo:** https://mohdanas-ctrl.github.io/churn-intelligence/

## Features
- Churn probability with Low / Medium / High risk band
- Plain-English reasoning for every influencing factor
- Priority retention actions and a long-term loyalty programme
- USD / INR currency switch and UPI payment option
- Runs fully in the browser, so no data leaves your device

## Models compared
Logistic Regression, Decision Tree, Random Forest and Gradient Boosting, evaluated with accuracy, precision, recall, F1-score and ROC-AUC.

## Run it
    pip install pandas numpy scikit-learn matplotlib joblib
    python churn_model.py
    python build_app.py

## Files
- index.html: the web app
- churn_model.py: training and evaluation pipeline
- build_app.py, template.html, model.json: export the model into the web app

Note: the demo uses synthetic Telco-style data for illustration. Retrain on real data for production use.

Made by **Mohd Anas**
