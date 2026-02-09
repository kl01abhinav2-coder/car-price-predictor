from flask import Flask, render_template, request
import numpy as np
import pickle
import datetime

app = Flask(__name__)
model = pickle.load(open('rfr_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Capture inputs and convert strictly to numbers
        # Ensure the HTML names match: Year, Present_Price, Kms_Driven, Fuel_Type, Transmission
        year = int(request.form['Year'])
        present_price = float(request.form['Present_Price'])
        kms_driven = int(request.form['Kms_Driven'])
        fuel_type = int(request.form['Fuel_Type'])
        transmission = int(request.form['Transmission'])
        
        # 2. Prevent futuristic years or impossible prices
        if year > 2024: year = 2024
        
        # 3. Calculate Age (Standardized feature for most car models)
        no_year = 2024 - year

        # 4. Standardizing the Feature Array (9-feature format for CarDekho)
        # Many models fail if the feature order is wrong. 
        # Check if your model expects [Price, Kms, Fuel_D, Fuel_P, Seller_I, Trans_M, Age, Owner1, Owner2]
        
        fuel_diesel = 1 if fuel_type == 1 else 0
        fuel_petrol = 1 if fuel_type == 0 else 0
        trans_manual = 1 if transmission == 0 else 0

        # Constructing the array with placeholder '0' for Seller and Owner to avoid high bias
        final_features = np.array([[
            present_price, 
            kms_driven, 
            fuel_diesel, 
            fuel_petrol, 
            0, # Seller_Individual (0 = Dealer)
            trans_manual, 
            no_year,
            0, # Owner placeholder 1
            0  # Owner placeholder 2
        ]])

        # 5. Prediction
        prediction = model.predict(final_features)
        
        # 6. Safety Check: If prediction is negative or absurdly high, cap it
        output = round(prediction[0], 2)
        if output < 0: output = 0.0
        if output > (present_price * 1.5): output = round(present_price * 0.8, 2) # Basic sanity logic

        return render_template('index.html', prediction_text=str(output))

    except Exception as e:
        print(f"Error Details: {e}")
        return render_template('index.html', prediction_text="Error in calculation")

if __name__ == "__main__":
    app.run(debug=True)