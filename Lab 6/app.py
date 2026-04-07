import cv2
import easyocr
from flask import Flask, render_template, request
import os
import numpy as np

app = Flask(__name__)

# Initialize the OCR reader (English)
reader = easyocr.Reader(['en'], gpu=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Ensure folders exist
            if not os.path.exists('static'):
                os.makedirs('static')
            if not os.path.exists('output'):
                os.makedirs('output')
                
            filepath = os.path.join('static', file.filename)
            file.save(filepath)
            
            # 1. Load and prepare image
            img = cv2.imread(filepath)
            if img is None:
                return "Error: Could not read image."
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 2. Find the Plate Location
            bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
            edged = cv2.Canny(bfilter, 30, 200) 
            
            keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(keypoints[0], key=cv2.contourArea, reverse=True)[:10]
            
            location = None
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 15, True)
                if 4 <= len(approx) <= 6: 
                    location = approx
                    break
            
            # 3. Processing & OCR
            result = []
            if location is not None:
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [location], 0, 255, -1)
                
                # Tight crop the image
                (x, y) = np.where(mask == 255)
                if len(x) > 0 and len(y) > 0:
                    (topx, topy) = (np.min(x), np.min(y))
                    (bottomx, bottomy) = (np.max(x), np.max(y))
                    cropped_image = gray[topx:bottomx+1, topy:bottomy+1]
                    
                    # --- SAVE TO OUTPUT FOLDER ---
                    output_filename = "plate_" + file.filename
                    output_path = os.path.join('output', output_filename)
                    cv2.imwrite(output_path, cropped_image)
                    # -----------------------------
                    
                    # Try OCR on the cropped plate
                    result = reader.readtext(cropped_image, detail=0)
            
            # 4. Fallback Logic: If no result or no location, use Adaptive Thresholding
            if not result:
                adaptive_thresh = cv2.adaptiveThreshold(gray, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
                result = reader.readtext(adaptive_thresh, detail=0)

            # 5. Smart Filter
            final_plates = [word for word in result if any(char.isdigit() for char in word)]
            
            if not final_plates and result:
                plate_text = " ".join(result)
            else:
                plate_text = " ".join(final_plates) if final_plates else "No Plate Detected"
            
            return render_template('index.html', text=plate_text, image=file.filename)
            
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)