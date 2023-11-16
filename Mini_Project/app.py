import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load your machine learning model and label encoders
model = joblib.load('randomforest.pkl')  # Replace 'your_model.pkl' with your model file
location_name_encoder = joblib.load('enoder2.pkl')  # Replace with your location_name label encoder
region_encoder = joblib.load('enoder3.pkl')  # Replace with your region label encoder
condition_text_encoder = joblib.load('enoder1.pkl')  # Replace with your condition_text label encoder


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        location_name = request.form['location_name']
        region = request.form['region']
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        temperature_celsius = float(request.form['temperature_celsius'])
        wind_kph = float(request.form['wind_kph'])
        pressure_mb = float(request.form['pressure_mb'])
        precip_mm = float(request.form['precip_mm'])
        humidity = int(request.form['humidity'])
        cloud = int(request.form['cloud'])
        visibility_km = int(request.form['visibility_km'])

        # Encode categorical features using label encoders
        location_name_encoded = location_name_encoder.transform([location_name])[0]
        region_encoded = region_encoder.transform([region])[0]

        # Create a DataFrame with the input features
        input_data = pd.DataFrame({
            'location_name': [location_name_encoded],
            'region': [region_encoded],
            'latitude': [latitude],
            'longitude': [longitude],
            'temperature_celsius': [temperature_celsius],
            'wind_kph': [wind_kph],
            'pressure_mb': [pressure_mb],
            'precip_mm': [precip_mm],
            'humidity': [humidity],
            'cloud': [cloud],
            'visibility_km': [visibility_km]
        })

        # Make a prediction using your machine learning model
        predicted_condition_text_encoded = model.predict(input_data)[0]

        # Decode the predicted condition_text
        predicted_condition_text = condition_text_encoder.inverse_transform([predicted_condition_text_encoded])[0]

        return render_template('result.html', predicted_condition_text=predicted_condition_text)

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
