from flask import Flask, render_template, request
import cv2
import os

app = Flask(__name__)
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = cv2.imread(filepath)

            if img is None:
                return "Error: Could not read image. Please upload JPG or PNG format."
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            smile_detected = False

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]

                smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

                if len(smiles) > 0:
                    smile_detected = True
                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)

            result_text = "Smiling 😊" if smile_detected else "Not Smiling 😐"

            cv2.imwrite(filepath, img)

            return render_template('index.html', image=filepath, result=result_text)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)