import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the class labels (COCO dataset includes car, truck, and motorbike)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load video
video_path = "C:/Users/DELL/Downloads/traffic_vdo.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize variables for counting
vehicle_count = {"car": 0, "truck": 0, "motorbike": 0}
tracking_objects = {}  # To track centroids of vehicles
track_id = 0

# Define the minimum confidence and NMS threshold
confidence_threshold = 0.5
nms_threshold = 0.4

def get_centroid(x, y, w, h):
    return (int(x + w / 2), int(y + h / 2))

def is_new_vehicle(centroid, tracked_objects):
    for obj_id, (centroid_tracked, _) in tracked_objects.items():
        distance = np.sqrt((centroid[0] - centroid_tracked[0]) ** 2 + (centroid[1] - centroid_tracked[1]) ** 2)
        if distance < 50:  # Consider it the same vehicle if centroids are within 50 pixels
            return obj_id
    return None

# Create a named window to enable resizing
cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vehicle Detection", 1280, 720)  # Set desired width and height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Prepare input for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Process detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                # Get bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        # Update tracking and counting
        for i in indices.flatten():  # Flatten to ensure single index access
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])

            # Track vehicles by centroid
            if label in ["car", "truck", "motorbike"]:
                centroid = get_centroid(x, y, w, h)
                vehicle_id = is_new_vehicle(centroid, tracking_objects)

                if vehicle_id is None:
                    # It's a new vehicle, assign an ID and count it
                    track_id += 1
                    tracking_objects[track_id] = (centroid, label)
                    vehicle_count[label] += 1
                else:
                    # Update the centroid of the existing vehicle
                    tracking_objects[vehicle_id] = (centroid, label)

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display vehicle counts on the video
    count_text = f"Cars: {vehicle_count['car']} | Trucks: {vehicle_count['truck']} | Bikes: {vehicle_count['motorbike']}"
    cv2.putText(frame, count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(1) == 27:  # Exit when ESC key is pressed
        break

cap.release()
cv2.destroyAllWindows()
