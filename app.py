import cv2
import pytesseract
import numpy as np
import time
import torch
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import threading

# Set up Tesseract executable path (update as needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Load a pre-trained model

# Initialize parameters
fps = 30  # Default frames per second
MAX_SPEED = 80  # Maximum speed limit for capping

# Function to detect vehicles
def detect_vehicles(frame):
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Bounding boxes with confidence, class, etc.
    return detections

# Improved function to recognize license plate using OCR
def recognize_plate(frame, bbox):
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    plate_img = frame[y1:y2, x1:x2]

    # Image pre-processing steps for better OCR accuracy
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise

    # Additional edge detection (using Canny) for clarity
    edges = cv2.Canny(gray, 100, 200)

    # Adaptive thresholding to isolate characters on the plate
    _, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tesseract configuration for license plate (only digits and letters)
    config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    # Pass pre-processed image to Tesseract for better accuracy
    plate_text = pytesseract.image_to_string(thresh, config=config).strip()

    return plate_text

# Function to estimate the speed of the vehicle
def estimate_speed(positions, timestamps):
    if len(positions) < 2:
        return 0

    delta_x = positions[-1][0] - positions[-2][0]
    delta_y = positions[-1][1] - positions[-2][1]
    distance = np.sqrt(delta_x**2 + delta_y**2)
    time_diff = timestamps[-1] - timestamps[-2]

    speed = distance / time_diff if time_diff > 0 else 0

    adjusted_distance_per_frame = 0.01  # Adjust based on your setup
    speed_kmh = speed * adjusted_distance_per_frame * fps * 3.6  # Convert to km/h
    return min(speed_kmh, MAX_SPEED)

# Function to detect vehicle color
def detect_color(frame, bbox):
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    vehicle_img = frame[y1:y2, x1:x2]
    hsv_img = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)

    COLOR_BOUNDS = {
        'Red': [(np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))],
        'Green': [(np.array([40, 100, 100]), np.array([80, 255, 255]))],
        'Blue': [(np.array([86, 100, 100]), np.array([126, 255, 255]))],
        'Yellow': [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
        'White': [(np.array([0, 0, 200]), np.array([180, 25, 255]))],
    }

    detected_color = 'Unknown'
    for color_name, bounds in COLOR_BOUNDS.items():
        for lower_bound, upper_bound in bounds:
            mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
            if cv2.countNonZero(mask) > 0:
                detected_color = color_name
                break
        if detected_color != 'Unknown':
            break

    return detected_color

# Main function to process the video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    positions = []
    timestamps = []

    while cap.isOpened():  # Keep reading frames till the video ends
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop when video ends

        height, width = frame.shape[:2]
        resized_frame = cv2.resize(frame, (1280, int(1280 * height / width)))

        detections = detect_vehicles(frame)

        for _, detection in detections.iterrows():
            if detection['confidence'] > 0.4 and detection['name'] == 'car':
                x1 = int(detection['xmin'] * 1280 / width)
                y1 = int(detection['ymin'] * height / height)
                x2 = int(detection['xmax'] * 1280 / width)
                y2 = int(detection['ymax'] * height / height)

                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                positions.append((center_x, center_y))
                timestamps.append(time.time())

                if len(positions) > 1:
                    speed = estimate_speed(positions, timestamps)
                    cv2.putText(resized_frame, f"Speed: {speed:.2f} km/h", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Improved license plate recognition
                plate_text = recognize_plate(resized_frame, [x1, y1, x2, y2])
                if plate_text:
                    cv2.putText(resized_frame, f"Plate: {plate_text}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                vehicle_color = detect_color(resized_frame, [x1, y1, x2, y2])
                cv2.putText(resized_frame, f"Color: {vehicle_color}", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Vehicle Detection", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to select video file
def select_video():
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if video_path:
        # Run the video processing in a separate thread to prevent GUI freezing
        threading.Thread(target=process_video, args=(video_path,)).start()
    else:
        messagebox.showwarning("No file selected", "Please select a video file to process.")

# Function to quit the application
def quit_app():
    cv2.destroyAllWindows()
    root.quit()

# Main GUI setup
def create_app():
    global root
    root = tk.Tk()
    root.title("autoEye")
    root.geometry("400x400")
    root.configure(bg='grey')

    # Load logo
    try:
        logo = Image.open("logo.jpeg")
        logo = logo.resize((300, 300))
        logo = ImageTk.PhotoImage(logo)

        logo_label = tk.Label(root, image=logo, bg='grey')
        logo_label.image = logo
        logo_label.pack(pady=10)
    except Exception as e:
        messagebox.showerror("Logo Load Error", f"Failed to load logo: {e}")

    label = tk.Label(root, text="Vehicle Detection, License Plate, and Speed Estimation", font=("Helvetica", 14), padx=10, pady=20, bg='grey')
    label.pack()

    btn_select = tk.Button(root, text="Select Video", command=select_video, font=("Helvetica", 12), padx=10, pady=10)
    btn_select.pack()

    btn_quit = tk.Button(root, text="Quit", command=quit_app, font=("Helvetica", 12), padx=10, pady=10)
    btn_quit.pack()

    root.mainloop()

if __name__ == '__main__':
    create_app()
