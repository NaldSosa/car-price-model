from flask import Flask, request, render_template
import pandas as pd
import pickle
import gzip

app = Flask(__name__)

# Load model
with gzip.open("model.pkl.gz", "rb") as f:
    model = pickle.load(f)

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html", prediction=None, years=None, prices=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form.to_dict()
        input_df = pd.DataFrame([form])

        # Convert numeric columns
        for col in ['Year', 'Owner', 'KM_Driven', 'Mileage', 'Engine_CC', 'Max_Power_BHP']:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # Apply encoders to categorical columns
        for col, encoder in encoders.items():
            input_df[col] = encoder.transform(input_df[col])

        # Predict price for the current year
        current_price = model.predict(input_df)[0]

        # Years to predict for
        years = list(range(2025, 2036))
        prices = []

        manuf_year = int(input_df.at[0, 'Year'])
        base_km = input_df.at[0, 'KM_Driven']

        for year in years:
            temp_df = input_df.copy()
            temp_df['Year'] = year

            # Simulate mileage increasing by 10,000 km per year since manufacture
            years_passed = year - manuf_year
            temp_df['KM_Driven'] = base_km + 10000 * years_passed

            # Predict price for this future year
            price = model.predict(temp_df)[0]
            prices.append(round(price, 2))

        return render_template("index.html",
                               prediction=f"${current_price:,.2f}",
                               years=years,
                               prices=prices)
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}", years=None, prices=None)

pass
