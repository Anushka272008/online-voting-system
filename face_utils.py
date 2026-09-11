import cv2
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

detector = cv2.FaceDetectorYN.create(
    os.path.join(BASE_DIR, "face_detection_yunet_2026may.onnx"),
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    os.path.join(BASE_DIR, "face_recognition_sface_2021dec_int8.onnx"),
    ""
)

def get_face_feature(frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    if faces is None or len(faces) != 1:
        return None

    crop = recognizer.alignCrop(frame, faces[0])
    return recognizer.feature(crop).flatten()

def compare_faces(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(1, -1)
    b = np.asarray(b, dtype=np.float32).reshape(1, -1)
    return float(recognizer.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))

def feature_to_json(f):
    return json.dumps(np.asarray(f).tolist())

def json_to_feature(x):
    return np.asarray(json.loads(x), dtype=np.float32)
