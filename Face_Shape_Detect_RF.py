import os
import joblib
import dlib
import math
import numpy as np
import cv2
import mediapipe as mp



# Calculate the Euclidean distance between two 3D points.
def calculate_distance(p1, p2):
    # p1 and p2 are assumed to be objects with attributes x, y, and z.
    # The distance is computed using the 3D Euclidean formula.
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# Calculate the angle (in degrees) at point p2 formed by the line segments p2-p1 and p2-p3.
def calculate_angle(p1, p2, p3):
    # a: distance between p2 and p3 (side opposite to p1)
    a = calculate_distance(p2, p3)
    # b: distance between p1 and p3 (side opposite to p2)
    b = calculate_distance(p1, p3)
    # c: distance between p1 and p2 (side opposite to p3)
    c = calculate_distance(p1, p2)
    # Use the cosine rule to calculate the angle at p2:
    # cos(angle) = (b^2 + c^2 - a^2) / (2 * b * c)
    angle_rad = math.acos((b**2 + c**2 - a**2) / (2 * b * c))
    # Convert the angle from radians to degrees.
    return math.degrees(angle_rad)

# Compute the area of the bounding rectangle that encloses a set of landmarks.
def bounding_rectangle_area(landmarks, indices):
    # Create an array of (x, y) coordinates for the landmarks at the specified indices.
    pts = np.array([(landmarks[i].x, landmarks[i].y) for i in indices])
    # Determine the minimum and maximum x and y values among these points.
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    # Calculate the area of the bounding rectangle.
    area = (x_max - x_min) * (y_max - y_min)
    return area

def extract_features(image):
    """
    Extracts facial features from an image, including distances, angles, and ratios.
    Uses MediaPipe to detect facial landmarks.
    """

    # Convert the image from BGR (OpenCV default) to RGB (required by MediaPipe)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize the MediaPipe FaceMesh model
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image)

        # If no face is detected, return None
        if not results.multi_face_landmarks:
            return None

        # Get the list of facial landmarks
        lm = results.multi_face_landmarks[0].landmark

        # Define key facial landmarks
        cheek_left  = lm[234]
        cheek_right = lm[454]
        nose_left   = lm[49]
        nose_right  = lm[279]
        eye_brow_left  = lm[70]
        eye_brow_right = lm[300]
        chin_left   = lm[172]
        chin_right  = lm[397]
        forehead_mid = lm[10]   # Middle of the forehead
        forehead_left = lm[109]  # Left forehead corner
        forehead_right = lm[338] # Right forehead corner
        chin_mid = lm[152]       # Middle of the chin

        # 1. Face Rectangularity
        face_area = bounding_rectangle_area(lm, [10, 109, 338, 234, 454, 172, 397, 152])
        bounding_rect_face = bounding_rectangle_area(lm, [10, 152, 234, 454])
        face_rectangularity = face_area / bounding_rect_face if bounding_rect_face > 0 else 0

        # 2. Middle Face Rectangularity
        middle_rect_area = bounding_rectangle_area(lm, [234, 454, 172, 397])
        face_middle_rectangularity = middle_rect_area / bounding_rect_face if bounding_rect_face > 0 else 0

        # 3. Forehead Rectangularity
        forehead_rect_area = bounding_rectangle_area(lm, [10, 109, 338])
        forehead_rectangularity = forehead_rect_area / bounding_rect_face if bounding_rect_face > 0 else 0

        # 4. Chin Angle
        chin_angle = calculate_angle(chin_left, chin_mid, chin_right)

        # 5. RBot = Lower face width / Middle face width
        lower_face_width = calculate_distance(chin_left, chin_right)
        middle_face_width = calculate_distance(cheek_left, cheek_right)
        RBot = lower_face_width / middle_face_width if middle_face_width > 0 else 0

        # 6. RTop = Forehead width / Middle face width
        forehead_width = calculate_distance(forehead_left, forehead_right)
        RTop = forehead_width / middle_face_width if middle_face_width > 0 else 0

        # 7. Difference between RTop and RBot
        RTop_RBot_diff = RTop - RBot

        # 8. fAR = Face width / Face height
        face_width = calculate_distance(cheek_left, cheek_right)
        face_height = calculate_distance(forehead_mid, chin_mid)
        fAR = face_width / face_height if face_height > 0 else 0

        # 9. Left cheek width
        left_cheek_distance = calculate_distance(cheek_left, nose_left)

        # 10. Right cheek width
        right_cheek_distance = calculate_distance(cheek_right, nose_right)

        # 11. Right cheek angle
        right_cheek_angle = calculate_angle(lm[152], lm[400], chin_right)

        # 12. Left cheek angle
        left_cheek_angle = calculate_angle(lm[152], lm[176], chin_left)

        # 13. Face length
        face_length = calculate_distance(lm[10], lm[152])

        # 14. Cheekbone width
        cheekbone_width = calculate_distance(lm[234], lm[454])

        # 15. Jawline width
        jawline_width = 2 * calculate_distance(lm[454], lm[152])

        # 16. Top jaw width
        top_jaw_distance = calculate_distance(nose_left, nose_right)

        # 17. Forehead width based on eyebrow distance
        forehead_distance = calculate_distance(eye_brow_left, eye_brow_right)

        # 18. Chin width
        chin_distance = calculate_distance(chin_left, chin_right)

        # List of extracted facial features
        features = [
            RBot,
            RTop,
            RTop_RBot_diff,
            cheekbone_width,
            chin_angle,
            chin_distance,
            fAR,
            face_length,
            face_rectangularity,
            forehead_distance,
            forehead_rectangularity,
            forehead_width,
            jawline_width,
            left_cheek_angle,
            left_cheek_distance,
            face_middle_rectangularity,
            right_cheek_angle,
            right_cheek_distance,
            top_jaw_distance
        ]

        return features
        


def detect_face_shape(image_path):
    """
    Predicts the face shape using a pre-trained Random Forest model.

    Parameters:
    - image_path (str): Path to the input image.

    Returns:
    - face_shape (str): Predicted face shape category.
    """

    # Load the pre-trained Random Forest model
    rf_model = joblib.load(fr"models/rf_model.pkl")

    # Load the label encoder used to convert numerical labels back to category names
    le = joblib.load(fr"models/label_encoder.pkl")

    # Read the input image
    image = cv2.imread(image_path)

    # Extract facial features from the image
    features = extract_features(image)

    # Predict the face shape using the Random Forest model
    prediction_num = rf_model.predict([features])[0]

    # Convert numerical prediction back to the corresponding face shape label
    face_shape = le.inverse_transform([prediction_num])[0]

    return face_shape



img_path= fr'FaceShape_Dataset\testing_set\Square\square (61).jpg'
# print(f'Predicted Face Shape: {detect_face_shape(img_path)}')