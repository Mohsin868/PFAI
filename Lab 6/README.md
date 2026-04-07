# 🚗 Vehicle Number Plate Recognition System

## 📌 Project Overview

This project is a **Vehicle Number Plate Recognition (VNPR) system** built using **Python, OpenCV, and Tesseract OCR**.
It detects a vehicle’s number plate from an image and extracts the text using Optical Character Recognition (OCR).

---

## 🎯 Objectives

* Detect number plates from vehicle images
* Extract alphanumeric characters from the plate
* Store extracted data in a CSV file
* Demonstrate basic computer vision techniques

---

## 🛠️ Technologies Used

* Python 🐍
* OpenCV (Computer Vision)
* NumPy
* Imutils
* Pytesseract (OCR)
* Pandas

---

## ⚙️ How It Works

1. **Image Input**

   * Load and resize the input image

2. **Preprocessing**

   * Convert image to grayscale
   * Apply bilateral filtering to remove noise

3. **Edge Detection**

   * Use Canny Edge Detection to highlight edges

4. **Contour Detection**

   * Find contours and sort them by area
   * Detect rectangular contours likely to be number plates

5. **Plate Extraction**

   * Crop the detected number plate region

6. **Text Recognition**

   * Use Tesseract OCR to extract text

7. **Data Storage**

   * Save detected number and timestamp into an output folder

---

## 📂 Project Structure

Lab 6/
│── app.py
│── static
│── index.html
│── README.md
│── output


---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Mohsin868/PFAI/tree/main/Lab%206
cd Lab 6
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install opencv-python imutils pytesseract numpy pandas
```

### 4️⃣ Install Tesseract OCR

Download from:
https://github.com/UB-Mannheim/tesseract/wiki

After installation, update path in code:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## ▶️ How to Run

```bash
python app.py
```

---

## 📊 Output

* Displays:

  * Original Image
  * Edges
  * Detected Plate Region

* Console Output:

```
Detected Number Plate: ABC1234
```

```

| date      | v_number |
| --------- | -------- |
| timestamp | ABC1234  |

---

## ⚠️ Limitations

* Works best with:

  * Clear images
  * Front-facing number plates
  * Good lighting conditions

* May fail with:

  * Blurry images
  * Low contrast plates
  * Complex backgrounds

---

## 🔮 Future Improvements

* Use YOLO for better plate detection
* Real-time camera detection
* Web app using Flask
* Improve OCR accuracy using preprocessing

---

## 👨‍💻 Author

Muhammad Mohsin Ahmad
BS Data Science – Superior University

---

## 📜 License

This project is for educational purposes only.
