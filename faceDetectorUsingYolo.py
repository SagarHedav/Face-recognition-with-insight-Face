import cv2
import numpy as np
import torch
import insightface
import time

# Load YOLOv5 face detector (replace 'yolov5s-face.pt' with your face model)
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)

# Load InsightFace for recognition
recognizer = insightface.app.FaceAnalysis(name='buffalo_l')
recognizer.prepare(ctx_id=0)

similarity_threshold = 0.6

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding_from_file(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Use YOLO for face detection
    results = yolo_model(img_rgb)
    detections = results.xyxy[0].cpu().numpy()
    if len(detections) > 0:
        x1, y1, x2, y2, conf, cls = detections[0]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        face_img = img_rgb[y1:y2, x1:x2]
        if face_img.size == 0:
            print("Detected face region is empty.")
            return None
        faces = recognizer.get(face_img)
        if faces:
            return faces[0].embedding
    print("No face detected in image.")
    return None

def detect_and_recognize(frame, known_embedding):
    # Use YOLO for face detection
    results = yolo_model(frame)
    detections = results.xyxy[0].cpu().numpy()
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            continue
        faces = recognizer.get(face_img)
        if faces:
            emb = faces[0].embedding
            sim = cosine_similarity(emb, known_embedding)
            if sim > similarity_threshold:
                return True, (x1, y1, x2, y2)
    return False, None

if __name__ == "__main__":
    known_name = "Sagar"
    sagar_embedding = get_embedding_from_file("public/sagar.jpg")
    if sagar_embedding is None:
        print("No face found in sagar.jpg")
        exit(1)

    video_capture = cv2.VideoCapture(0)
    recognized = False
    start_time = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not recognized:
            start_time = time.time()
            recognized, bbox = detect_and_recognize(frame_rgb, sagar_embedding)
            if recognized:
                end_time = time.time()
                print(f"Time required to recognize: {end_time - start_time:.2f} seconds")
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, known_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
                cv2.imshow("Video", frame)
                cv2.waitKey(2000)  # Show result for 2 seconds
                break
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()