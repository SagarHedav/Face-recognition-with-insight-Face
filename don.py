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
    results = []
    for face in faces:
        emb = face.embedding
        sim = cosine_similarity(emb, known_embedding)
        name = known_name if sim > similarity_threshold else "Unknown"
        bbox = face.bbox.astype(int)
        results.append((name, bbox))
    return results

if __name__ == "__main__":
    known_name = "Sagar"
    sagar_embedding = get_embedding_from_file("public/sagar.jpg")
    if sagar_embedding is None:
        print("No face found in sagar.jpg")
        exit(1)

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detect_and_recognize(frame_rgb, sagar_embedding, known_name)
        
        for name, bbox in results:
            # Draw rectangle around face
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Add name text above the face
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.8
            font_thickness = 2
            text_size = cv2.getTextSize(name, font, font_scale, font_thickness)[0]
            text_x = bbox[0] + (bbox[2] - bbox[0] - text_size[0]) // 2
            text_y = bbox[1] - 10
            
            # Draw text background
            cv2.rectangle(frame, 
                        (text_x - 5, text_y - text_size[1] - 5),
                        (text_x + text_size[0] + 5, text_y + 5),
                        (0, 255, 0), 
                        -1)
            
            # Draw text
            cv2.putText(frame, name, (text_x, text_y), 
                        font, font_scale, (0, 0, 0), font_thickness)

        cv2.imshow("Video", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
