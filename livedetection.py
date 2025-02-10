import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity

# Initialize models and load necessary files
facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings.npz")
known_embeddings = faces_embeddings['arr_0']  # Load stored embeddings
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

# Load the YOLOv8 face detection model
face_detector = YOLO('yolov8l-face.pt')

# Load the SVM model for face recognition
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

def calculate_confidence(embedding, known_embeddings):
    """
    Calculate confidence score based on similarity to known faces
    
    Args:
        embedding: Face embedding of the current face
        known_embeddings: Embeddings of all known faces
        
    Returns:
        float: Confidence score between 0 and 1
    """
    # Calculate cosine similarities between current face and all known faces
    similarities = cosine_similarity(embedding, known_embeddings)
    # Get the highest similarity score
    max_similarity = np.max(similarities)
    return max_similarity

def process_yolo_detection(results, frame, confidence_threshold=0.70):
    """
    Process YOLOv8 detection results and perform face recognition with unknown detection
    
    Args:
        results: YOLOv8 detection results
        frame: Original frame from video capture
        confidence_threshold: Minimum confidence to classify as known face
        
    Returns:
        frame: Processed frame with detections and recognition results
    """
    if len(results) > 0:
        result = results[0]
        rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        for box in result.boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Extract and process face
            face_roi = rgb_img[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue
                
            face_roi = cv.resize(face_roi, (160, 160))
            face_roi = np.expand_dims(face_roi, axis=0)
            
            # Get face embedding
            face_embedding = facenet.embeddings(face_roi)
            
            # Calculate confidence score
            confidence = calculate_confidence(face_embedding, known_embeddings)
            
            # Predict identity or mark as unknown
            if confidence >= confidence_threshold:
                face_name = model.predict(face_embedding)
                final_name = encoder.inverse_transform(face_name)[0]
                # Display with confidence
                label = f"{final_name} ({confidence:.2%})"
                color = (0, 255, 0)  # Green for known faces
            
                # Draw results
                cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv.putText(frame, label, (x1, y1-10), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, 
                        cv.LINE_AA)
    
    return frame

# Initialize video capture
cap = cv.VideoCapture(0)

# Main loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # Perform face detection
    results = face_detector.predict(frame, conf=0.5)
    
    # Process detections with unknown detection
    frame = process_yolo_detection(results, frame)
    
    # Display output
    cv.imshow("Face Recognition:", frame)
    
    if cv.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release()
cv.destroyAllWindows()