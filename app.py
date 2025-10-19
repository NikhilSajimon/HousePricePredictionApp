from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained machine learning pipeline
try:
    with open('houseprice_pred.pkl', 'rb') as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    print("Error: 'houseprice_pred.pkl' not found. Make sure the model file is in the same directory.")
    pipeline = None

# Define the exact order of columns as expected by the trained model
# IMPORTANT: Added 'Lot Area (in Sqft)' as it was requested in the form.
# Ensure your saved model ('houseprice_pred.pkl') was trained with this feature.
MODEL_COLUMNS = [
    'No of Bedrooms', 'No of Bathrooms', 'No of Floors', 'Waterfront View',
    'Condition of the House', 'Overall Grade', 'Basement Area (in Sqft)',
    'Age of House (in Years)', 'Latitude', 'Longitude',
    'Living Area after Renovation (in Sqft)',
    'Lot Area after Renovation (in Sqft)',
    'Area of the House from Basement (in Sqft)'
]

@app.route('/')
def home():
    """Renders the first page of the form for basic house info."""
    return render_template('page1.html')

@app.route('/page2', methods=['POST'])
def page2():
    """Renders the second page for condition and view info, carrying over data from page 1."""
    # Collect data from the first form
    data_page1 = {
        'bedrooms': request.form['bedrooms'],
        'bathrooms': request.form['bathrooms'],
        'floors': request.form['floors']
    }
    return render_template('page2.html', data_page1=data_page1)

@app.route('/page3', methods=['POST'])
def page3():
    """Renders the third page for area and location, carrying over all previous data."""
    # Collect data from previous forms
    data_page2 = {
        'bedrooms': request.form['bedrooms'],
        'bathrooms': request.form['bathrooms'],
        'floors': request.form['floors'],
        'waterfront': request.form['waterfront'],
        'condition': request.form['condition'],
        'grade': request.form['grade'],
        'age': request.form['age']
    }
    return render_template('page3.html', data_page2=data_page2)

@app.route('/predict', methods=['POST'])
def predict():
    """Processes the complete form data, makes a prediction, and displays the result."""
    if not pipeline:
        return "Model not loaded. Please check the server logs.", 500

    try:
        # Collect all data from the final form submission
        form_data = request.form

        # Create a dictionary with all the required features, converting types
        features = {
            'No of Bedrooms': float(form_data['bedrooms']),
            'No of Bathrooms': float(form_data['bathrooms']),
            'No of Floors': float(form_data['floors']),
            'Waterfront View': form_data['waterfront'],
            'Condition of the House': form_data['condition'],
            'Overall Grade': int(form_data['grade']),
            'Area of the House from Basement (in Sqft)': float(form_data['area_from_basement']),
            'Basement Area (in Sqft)': float(form_data['basement_area']),
            'Age of House (in Years)': int(form_data['age']),
            'Latitude': float(form_data['latitude']),
            'Longitude': float(form_data['longitude']),
            'Living Area after Renovation (in Sqft)': float(form_data['living_area_renov']),
            'Lot Area after Renovation (in Sqft)': float(form_data['lot_area'])
        }

        # Create a DataFrame in the exact order the model expects
        input_df = pd.DataFrame([features], columns=MODEL_COLUMNS)

        # Make the prediction
        prediction = pipeline.predict(input_df)
        predicted_price = prediction[0]

        # Format the price for a user-friendly display
        formatted_price = f"${predicted_price:,.2f}"

        return render_template('result.html', price=formatted_price)
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return f"An error occurred: {e}", 400

if __name__ == '__main__':
    app.run(debug=True)

