import requests
from flask import Flask, render_template, request

app = Flask(__name__)

# 1. Correct Setup
API_URL = "https://api.api-ninjas.com/v1/cars?model={}"
API_KEY = "4aJrjveRuwDuT07i3yuHkd9FlexqPjNKwVXuEAHw"  # Your key is now in the right place!

@app.route('/', methods=['GET', 'POST'])
def vehicle_info():
    car_data = None
    error_msg = None
    kml = None

    if request.method == 'POST':
        model_name = request.form.get('car_model')
        if model_name:
            response = requests.get(API_URL.format(model_name), headers={'X-Api-Key': API_KEY})
            if response.status_code == 200:
                data = response.json()
                if data:
                    car_data = data[0]
                    # FIX: Force city_mpg to be a float before multiplying
                    try:
                        mpg_val = float(car_data.get('city_mpg', 0))
                        kml = round(mpg_val * 0.425, 1)
                    except (ValueError, TypeError):
                        kml = "N/A"
                else:
                    error_msg = "Vehicle not found. Try 'Civic' or 'Sportage'."
            else:
                error_msg = "API Connection Error."

    return render_template('vehicle.html', car=car_data, error=error_msg, kml=kml)

if __name__ == "__main__":
    app.run(debug=True)