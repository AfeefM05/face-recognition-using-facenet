This uses yolov8l-face (pretrained using WIDERFACE dataset) to detect faces
The detected faces are then cropped ,resized then passed onto FaceNet to extract features
Then SVM is trained to predict the accurate image with existing images

Real Time Live webcam input is taken and preprocessed for face detection and detected faces are displayed on the video feed.

yolov8l-face.pt is large so i couldnt upload it heres the github repo kindly download - https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8l-face.pt
Also the required libraries are given in requirements.txt kindly use the command in terminal to install - pip install -r requirements.txt
Details explanation of code is given as comments inside the files.

Thank You.
