import requests
import random
from flask import Flask, render_template, request

app = Flask(__name__)

# --- CONFIGURATION ---
# Replace with your actual API Key from API Ninjas
API_KEY = '4aJrjveRuwDuT07i3yuHkd9FlexqPjNKwVXuEAHw'
API_URL = 'https://api.api-ninjas.com/v1/cars?model={}'

# List for the "Dynamic" Random Feature (Lab 8 requirement)
RANDOM_CARS = ['Land Cruiser', 'Civic', 'Corolla', 'Vitz', 'Sportage', 'Mustang', 'Camry', 'Hilux']

@app.route('/', methods=['GET', 'POST'])
def vehicle_info():
    car_data = None
    error_msg = None
    kml = None
    model_query = None

    if request.method == 'POST':
        # 1. DYNAMIC INPUT: Check if user clicked 'Scan' or 'Random'
        if 'random_btn' in request.form:
            model_query = random.choice(RANDOM_CARS)
        else:
            model_query = request.form.get('car_model')

        if model_query:
            # 2. BACK-END: Fetch data from External API
            response = requests.get(API_URL.format(model_query), headers={'X-Api-Key': API_KEY})
            
            if response.status_code == 200:
                results = response.json()
                if results:
                    car_data = results[0]
                    
                    # 3. DATA PROCESSING: Convert MPG to KM/L (Local Relevance)
                    # We use a try-except to avoid the 'TypeError' if API sends a string
                    try:
                        mpg_val = float(car_data.get('city_mpg', 0))
                        kml = round(mpg_val * 0.425, 1)
                    except (ValueError, TypeError):
                        kml = "N/A"
                else:
                    error_msg = f"No records found for '{model_query}'. Try another model."
            else:
                error_msg = f"Server Error (Status: {response.status_code}). Please check API Key."

    # 4. FRONT-END RENDERING: Pass all variables to the template
    return render_template('index.html', car=car_data, error=error_msg, kml=kml, searched_name=model_query)

if __name__ == "__main__":
    app.run(debug=True)