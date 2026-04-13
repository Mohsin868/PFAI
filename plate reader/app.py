from flask import Flask, request, render_template
from ultralytics import YOLO
import cv2
import easyocr
import os

app = Flask(__name__)

# --- Configuration & Setup ---
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/output'

for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load optimized model
model = YOLO('best_openvino_model/', task='detect') 
reader = easyocr.Reader(['en'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400
        
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    # 1. YOLO Detects the plate
    results = model(path)
    
    # 2. Setup variables
    img = cv2.imread(path)
    plate_text = "No Plate Detected"
    crop_path = os.path.join(RESULT_FOLDER, 'cropped_plate.jpg')
    
    # 3. Process Detections
    for result in results:
        for box in result.boxes:
            # Get coordinates and crop
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = img[y1:y2, x1:x2]
            
            # --- IMPROVE OCR ACCURACY ---
            # Pre-processing: Grayscale helps EasyOCR see lines better
            gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            
            # 4. Read with EasyOCR
            ocr_result = reader.readtext(gray_plate)
            
            if ocr_result:
                # FIX: This joins "PUNJAB" and the "NUMBER" into one string
                # res[1] is the text detected in each block
                all_text_found = [res[1] for res in ocr_result]
                plate_text = " ".join(all_text_found)
            
            # Save the image for the UI
            cv2.imwrite(crop_path, plate_crop)
            
            # Break after the first valid plate found
            break 
            
    return render_template('index.html', text=plate_text, cropped_img=crop_path)

if __name__ == '__main__':
    app.run(debug=True)