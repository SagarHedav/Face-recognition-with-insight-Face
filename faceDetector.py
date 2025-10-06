import cv2
import insightface
import numpy as np
import time

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
    faces = recognizer.get(img_rgb)
    if faces:
        face = faces[0]
        emb = face.embedding
        return emb
    print("No face detected in image.")
    return None

def detect_and_recognize(frame, known_embedding, known_name):
    faces = recognizer.get(frame)
    print(faces)
    for face in faces:
        emb = face.embedding
        sim = cosine_similarity(emb, known_embedding)
        if sim > similarity_threshold:
            return True, known_name
    return False, None

if __name__ == "__main__":
    known_name = "Sagar"
    sagar_embedding = get_embedding_from_file("public/ronak.jpeg")
    if sagar_embedding is None:
        print("No face found in sagar.jpg")
        exit(1)

    video_capture = cv2.VideoCapture(0)
    start_time = None
    recognized = False

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not recognized:
            start_time = time.time()
            recognized, recognized_name = detect_and_recognize(frame_rgb, sagar_embedding, known_name)
            if recognized:
                end_time = time.time()
                print(f"Recognized: {recognized_name}")
                print(f"Time required to recognize: {end_time - start_time:.2f} seconds")
                break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow("Video", frame)

    video_capture.release()
    cv2.destroyAllWindows()