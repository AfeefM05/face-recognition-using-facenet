import cv2 as cv
import numpy as np
import os
from ultralytics import YOLO
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import pickle
import time
from threading import Thread
from queue import Queue
import torch
from concurrent.futures import ThreadPoolExecutor
import cupy as cp  # For GPU operations
from sklearn.metrics.pairwise import cosine_similarity

class CCTVFaceRecognition:
    def __init__(self, model_path="svm_model_160x160.pkl", 
                 embeddings_path="faces_embeddings.npz",
                 confidence_threshold=0.80,
                 detection_frequency=3,
                 max_processing_threads=10
                 ):
        """
        Initialize CCTV Face Recognition system with GPU optimization
        
        Args:
            model_path: Path to trained SVM model
            embeddings_path: Path to face embeddings
            confidence_threshold: Threshold for unknown face detection
            detection_frequency: Process every nth frame
            max_processing_threads: Maximum number of parallel processing threads
        """
        # Initialize models
        self.face_detector = YOLO('yolov8l-face.pt')
        self.facenet = FaceNet()
        self.detection_frequency = detection_frequency
        self.confidence_threshold = confidence_threshold
        self.max_processing_threads = max_processing_threads
        
        # Initialize queues
        self.frame_queue = Queue(maxsize=128)
        self.result_queue = Queue(maxsize=128)
        self.embedding_queue = Queue(maxsize=64)
        
        # Load embeddings and model
        self.load_recognition_model(model_path, embeddings_path)
        
        # Initialize thread pool
        self.executor = ThreadPoolExecutor(max_workers=max_processing_threads)
        
        # Performance monitoring
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.known_embeddings_gpu = cp.array(self.known_embeddings)
        
        # Batch processing setup
        self.batch_size = 8
        self.face_batch = []
        self.coords_batch = []

    def load_recognition_model(self, model_path, embeddings_path):
        """Load and prepare recognition models and data"""
        faces_embeddings = np.load(embeddings_path)
        self.known_embeddings = faces_embeddings['arr_0']
        Y = faces_embeddings['arr_1']
        
        self.encoder = LabelEncoder()
        self.encoder.fit(Y)
        
        self.recognition_model = pickle.load(open(model_path, 'rb'))

    def gpu_cosine_similarity(self, embedding):
        """Calculate cosine similarity on GPU"""
        if torch.cuda.is_available():
            # Convert embedding to CuPy array
            embedding_gpu = cp.array(embedding)
            
            # Calculate dot product
            dot_product = cp.dot(self.known_embeddings_gpu, embedding_gpu.T)
            
            # Calculate magnitudes
            embedding_norm = cp.linalg.norm(embedding_gpu)
            known_norms = cp.linalg.norm(self.known_embeddings_gpu, axis=1)
            
            # Calculate similarity
            similarities = dot_product / (embedding_norm * known_norms)
            
            return cp.asnumpy(similarities)
        else:
            # Fallback to CPU if GPU is not available
            return cosine_similarity(embedding, self.known_embeddings)[0]

    def process_face_batch(self, face_batch, coords_batch):
        """Process a batch of faces"""
        if not face_batch:
            return []
        
        # Prepare batch for FaceNet
        face_arrays = np.array(face_batch)
        embeddings = self.facenet.embeddings(face_arrays)
        
        results = []
        for embedding, (x1, y1, x2, y2) in zip(embeddings, coords_batch):
            # Calculate similarity using GPU
            similarities = self.gpu_cosine_similarity(embedding.reshape(1, -1))
            confidence = np.max(similarities)
            
            if confidence >= self.confidence_threshold:
                face_name = self.recognition_model.predict(embedding.reshape(1, -1))
                name = self.encoder.inverse_transform(face_name)[0]
                color = (0, 255, 0)
                results.append((name, confidence, color, (x1, y1, x2, y2)))
        
        return results

    def process_frame(self, frame):
        """Process a single frame for face detection and recognition"""
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Detect faces using YOLOv8
        results = self.face_detector.predict(rgb_frame, conf=0.5)
        
        if len(results) > 0:
            result = results[0]
            
            # Clear batches
            self.face_batch = []
            self.coords_batch = []
            
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                face_roi = rgb_frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue
                
                # Resize face
                face_roi = cv.resize(face_roi, (160, 160))
                
                # Add to batch
                self.face_batch.append(face_roi)
                self.coords_batch.append((x1, y1, x2, y2))
                
                # Process batch if full
                if len(self.face_batch) >= self.batch_size:
                    results = self.process_face_batch(self.face_batch, self.coords_batch)
                    self.draw_results(frame, results)
                    self.face_batch = []
                    self.coords_batch = []
            
            # Process remaining faces
            if self.face_batch:
                results = self.process_face_batch(self.face_batch, self.coords_batch)
                self.draw_results(frame, results)
        
        # Calculate and display FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        self.fps = self.frame_count / elapsed_time
        
        cv.putText(frame, f"FPS: {self.fps:.2f}", (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

    def draw_results(self, frame, results):
        """Draw detection results on frame"""
        for name, confidence, color, (x1, y1, x2, y2) in results:
            label = f"{name} ({confidence:.2%})"
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv.putText(frame, label, (x1, y1-10),
                      cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
                      cv.LINE_AA)

    def process_frames_thread(self):
        """Thread function for processing frames"""
        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break
            processed_frame = self.process_frame(frame)
            self.result_queue.put(processed_frame)

    def run_cctv(self, source=0):
        """Run CCTV face recognition system"""
        # Start processing threads
        processing_threads = [Thread(target=self.process_frames_thread) 
                            for _ in range(self.max_processing_threads)]
        for thread in processing_threads:
            thread.start()
        
        cap = cv.VideoCapture(source)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.detection_frequency == 0:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
            
            if not self.result_queue.empty():
                processed_frame = self.result_queue.get()
                cv.imshow("CCTV Face Recognition", processed_frame)
            
            frame_count += 1
            
            if cv.waitKey(1) & 0xFF in (ord('q'), 27):
                break
        
        # Cleanup
        for _ in range(self.max_processing_threads):
            self.frame_queue.put(None)
        for thread in processing_threads:
            thread.join()
        
        self.executor.shutdown()
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    cctv_system = CCTVFaceRecognition()
    cctv_system.run_cctv(0)
