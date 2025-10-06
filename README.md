# Face-recognition-with-insight-Face

A Python-based real-time face recognition system that uses computer vision to detect and identify faces through your webcam. The system can recognize known faces and label unknown faces in real-time with a clean visual interface.

## Features

- Real-time face detection and recognition
- Visual feedback with bounding boxes around detected faces
- Live labeling of recognized individuals
- Clean text overlay showing recognition results
- Unknown face detection and labeling
- Easy-to-use interface with webcam integration

## Technical Stack

- Python 3.x
- OpenCV (cv2) for video capture and image processing
- InsightFace for face detection and recognition
- NumPy for numerical computations

## How It Works

The system uses a two-step process:
1. First, it loads a known face image and creates an embedding (a mathematical representation of the face)
2. Then it continuously:
   - Captures video from the webcam
   - Detects faces in each frame
   - Compares detected faces with the known face embedding
   - Displays results in real-time with visual indicators

## Installation

```bash
pip install opencv-python insightface numpy
```

## Usage

1. Replace `public/sagar.jpg` with your reference image
2. Update the `known_name` variable with the person's name
3. Run the script:
```bash
python don.py
```
4. Press 'q' to quit the application

## Performance

- Uses efficient cosine similarity for face matching
- Real-time processing with minimal lag
- Configurable similarity threshold (currently set to 0.6)

## Future Improvements

- Support for multiple known faces
- Face recognition statistics and logging
- Enhanced UI with additional information
- Support for different video sources
- Face tracking capabilities

## License

[Your chosen license]

## Contributors

[Your name]

Feel free to contribute to this project by submitting pull requests or reporting issues.

##Demon 
<img width="1710" height="1071" alt="Screenshot 2025-10-06 at 4 00 05 PM" src="https://github.com/user-attachments/assets/5b0f4827-de05-4cc5-a569-36c3a48d5bce" />
<img width="1445" height="1037" alt="Screenshot 2025-10-06 at 3 59 23 PM" src="https://github.com/user-attachments/assets/7a86e3fb-08a3-47b0-ae82-36da17ad2354" />


