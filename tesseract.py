import cv2
import pytesseract
import numpy as np
import torch
from torchvision import transforms

# Set up Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLOv5 Model (you can use a fine-tuned model for license plates)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Replace 'best.pt' with your custom-trained model path

# Function to preprocess image for better OCR performance
def preprocess_for_ocr(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    # Apply adaptive thresholding or simple binary thresholding to improve contrast
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

# Function to detect and extract number plates
def detect_and_read_number_plate(image):
    # Inference on the image using YOLOv5
    results = model(image)

    # Extract bounding boxes and class labels
    for det in results.xyxy[0]:
        xmin, ymin, xmax, ymax, conf, cls = det
        
        # Assuming the class corresponding to number plates is detected (this depends on your model training)
        # Extract the region of interest (ROI) - the number plate
        plate_img = image[int(ymin):int(ymax), int(xmin):int(xmax)]

        # Preprocess the number plate image for OCR
        preprocessed_plate = preprocess_for_ocr(plate_img)

        # Use Tesseract OCR to read the number plate
        custom_config = r'--oem 3 --psm 6'  # OEM 3: Default, PSM 6: Assume a single block of text
        number_plate_text = pytesseract.image_to_string(preprocessed_plate, config=custom_config)

        print("Detected Number Plate:", number_plate_text.strip())

# Example usage
image_path = 'car_image.jpg'  # Replace with your image path
image = cv2.imread(image_path)

detect_and_read_number_plate(image)
