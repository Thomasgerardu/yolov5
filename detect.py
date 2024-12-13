from ultralytics import YOLO
import os
import cv2
import numpy as np

# Load your trained YOLO model
model = YOLO(r"runs\detect\train4\weights\best.pt")

# Define input and output folders
image_folder = r"C:\Github\surfconditions_database\data_heartbeach"
output_folder = r"C:\Github\yolov5\detected_vuurtoren"
os.makedirs(output_folder, exist_ok=True)

# Define the class name to check
target_class = "vuurtoren"

# Loop through all images in the folder
for image_file in os.listdir(image_folder)[:150]:
    if image_file.endswith((".jpg", ".png", ".jpeg")):  # Check for valid image files
        image_path = os.path.join(image_folder, image_file)
        
        # Run inference on the image
        results = model(image_path)
        
        # Check if the target class is detected
        detected = False
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                if model.names[class_id] == target_class:
                    detected = True
                    break
        
        # Save the image only if 'vuurtoren' is detected
        if detected:
            # Load the original image
            original_img = cv2.imread(image_path)  # OpenCV loads images in BGR format
            
            # Save the original image (no bounding boxes or scores added)
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, original_img)  # Save the original image