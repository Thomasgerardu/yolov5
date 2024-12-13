import yt_dlp
import cv2
from ultralytics import YOLO

# Load your YOLO model
model = YOLO("runs/detect/train3/weights/best.pt")

# YouTube video URL (replace with your actual stream URL)
youtube_url = "https://youtu.be/8nn9fAr9LCE"

# Use yt_dlp to get the direct stream URL
ydl_opts = {
    'format': 'best',  # Get the best available quality
    'quiet': True,
}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(youtube_url, download=False)
    stream_url = info_dict['url']

# Open the YouTube stream with OpenCV
cap = cv2.VideoCapture(stream_url)

# Define the target class to check
target_class = "vuurtoren"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Stream ended or cannot read frame.")
        break

    # Convert the frame to RGB (YOLO expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO inference on the frame
    results = model(frame_rgb)

    # Check if the target class is detected
    detected = False
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if model.names[class_id] == target_class:
                detected = True
                break

    # Display the annotated frame
    annotated_frame = results[0].plot()  # Annotated frame with detections
    cv2.imshow("YOLO YouTube Stream", annotated_frame)

    # Print detection status
    if detected:
        print(f"'{target_class}' detected in the stream!")
    else:
        print(f"'{target_class}' NOT detected in the stream.")

    # Press 'q' to quit the stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
