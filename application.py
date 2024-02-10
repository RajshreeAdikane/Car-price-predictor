import pickle
# from pyexpat import model
from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
# from sklearn.metrics import r2_score

app = Flask(__name__)
car = pd.read_csv("Cleaned_Car_data.csv")
cors=CORS(app)
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique())
    fuel_types = sorted(car['fuel_type'].unique())
    
    kilo_driven = None  # Initialize kilo_driven
    
    if request.method == 'POST':
        kilo_driven = request.form.get('kilo_driven')
    
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types, kilo_driven=kilo_driven)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        company = request.form.get('company')
        car_model = request.form.get('car_models')  # Change variable name to car_model
        year = request.form.get('year')
        fuel_type = request.form.get('fuel_type')  # Fix variable name typo
        kilo_driven = request.form.get('kilo_driven')

        # Ensure all necessary fields are provided
        if company and car_model and year and fuel_type and kilo_driven:
            # Assuming your dataset columns are different, modify this part accordingly
            prediction = model.predict(pd.DataFrame({'company': [company],
                                                     'name': [car_model],
                                                     'year': [year],
                                                     'fuel_type': [fuel_type],
                                                     'kms_driven': [kilo_driven]}))
            return str(np.round(prediction[0], 2))
        else:
            return "Error: Missing required fields for prediction."
    else:
        return "Error: Invalid request method."

if __name__ == "__main__":
    app.run(debug=True)
