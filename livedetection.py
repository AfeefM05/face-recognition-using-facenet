import cv2 as cv
import numpy as np
import os
from ultralytics import YOLO
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import time
from threading import Thread
from queue import Queue

class CCTVFaceRecognition:
    def __init__(self, model_path="svm_model_160x160.pkl", 
                 embeddings_path="faces_embeddings.npz",
                 confidence_threshold=0.80,
                 detection_frequency=3,
                 cooldown_period=10):  # Cooldown period in seconds
        """
        Initialize CCTV Face Recognition system
        
        Args:
            model_path: Path to trained SVM model
            embeddings_path: Path to face embeddings
            confidence_threshold: Threshold for unknown face detection
            detection_frequency: Process every nth frame
            cooldown_period: Minimum time before updating detection timestamp
        """
        # Initialize models
        self.face_detector = YOLO('yolov8l-face.pt')
        self.facenet = FaceNet()
        self.detection_frequency = detection_frequency
        self.confidence_threshold = confidence_threshold
        self.cooldown_period = cooldown_period  # Cooldown time in seconds

        # Load embeddings and model
        self.load_recognition_model(model_path, embeddings_path)

        # Dictionary to track last detection timestamps
        self.last_seen = {}

        # Initialize frame processing queue
        self.frame_queue = Queue(maxsize=128)
        self.result_queue = Queue(maxsize=128)

        # Performance monitoring
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def load_recognition_model(self, model_path, embeddings_path):
        """Load and prepare recognition models and data"""
        # Load face embeddings and labels
        faces_embeddings = np.load(embeddings_path)
        self.known_embeddings = faces_embeddings['arr_0']
        Y = faces_embeddings['arr_1']

        # Prepare label encoder
        self.encoder = LabelEncoder()
        self.encoder.fit(Y)

        # Load SVM model
        self.recognition_model = pickle.load(open(model_path, 'rb'))

    def process_frame(self, frame):
        """Process a single frame for face detection and recognition"""
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Detect faces using YOLOv8
        results = self.face_detector.predict(rgb_frame, conf=0.5)

        if len(results) > 0:
            result = results[0]
            for box in result.boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Extract and process face
                face_roi = rgb_frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue

                # Resize and get embedding
                face_roi = cv.resize(face_roi, (160, 160))
                face_roi = np.expand_dims(face_roi, axis=0)
                face_embedding = self.facenet.embeddings(face_roi)

                # Calculate similarity
                similarities = cosine_similarity(face_embedding, self.known_embeddings)
                confidence = np.max(similarities)

                # Determine identity
                if confidence >= self.confidence_threshold:
                    face_name = self.recognition_model.predict(face_embedding)
                    name = self.encoder.inverse_transform(face_name)[0]
                    color = (0, 255, 0)  # Green

                    # **Check timestamp to update detection**
                    current_time = time.time()
                    if name in self.last_seen:
                        time_since_last_seen = current_time - self.last_seen[name]
                        if time_since_last_seen < self.cooldown_period:
                            name = f"{name} (Recently Seen)"  # Mark as recently seen
                        else:
                            self.last_seen[name] = current_time  # Update timestamp
                    else:
                        self.last_seen[name] = current_time  # First-time detection

                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red

                # Draw the rectangle and put the name
                label = f"{name} ({confidence:.2%})"
                cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv.putText(frame, label, (x1, y1-10),
                          cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
                          cv.LINE_AA)

        # Calculate FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        self.fps = self.frame_count / elapsed_time

        # Add FPS to frame
        cv.putText(frame, f"FPS: {self.fps:.2f}", (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def process_frames_thread(self):
        """Thread function for processing frames"""
        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break
            processed_frame = self.process_frame(frame)
            self.result_queue.put(processed_frame)

    def run_cctv(self, source=0):
        """
        Run CCTV face recognition system
        
        Args:
            source: Camera source (0 for default webcam, rtsp:// URL for IP camera)
        """
        # Start processing thread
        processing_thread = Thread(target=self.process_frames_thread)
        processing_thread.start()

        # Initialize video capture
        cap = cv.VideoCapture(source)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every nth frame
            if frame_count % self.detection_frequency == 0:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())

            # Display processed frame if available
            if not self.result_queue.empty():
                processed_frame = self.result_queue.get()
                cv.imshow("CCTV Face Recognition", processed_frame)

            frame_count += 1

            if cv.waitKey(1) & 0xFF in (ord('q'), 27):
                break

        # Cleanup
        self.frame_queue.put(None)
        processing_thread.join()
        cap.release()
        cv.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    cctv_system = CCTVFaceRecognition(detection_frequency=15, cooldown_period=10)
    # For IP camera, use: cctv_system.run_cctv("rtsp://username:password@ip_address:port/stream")
    cctv_system.run_cctv("http://192.168.137.65:4747/video")  # For webcam
