import os
import cv2
import numpy as np
import mediapipe as mp
import math
import pandas as pd
import dlib

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def calculate_angle(p1, p2, p3):
    a = calculate_distance(p2, p3)
    b = calculate_distance(p1, p3)
    c = calculate_distance(p1, p2)
    angle_rad = math.acos((b**2 + c**2 - a**2) / (2 * b * c))
    return math.degrees(angle_rad)


def bounding_rectangle_area(landmarks, indices):
    pts = np.array([(landmarks[i].x, landmarks[i].y) for i in indices])
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    area = (x_max - x_min) * (y_max - y_min)
    return area

def process_image(image):
    
    
    image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        
        results = face_mesh.process(image)
        if not results.multi_face_landmarks:
            return {}
        face_landmarks = results.multi_face_landmarks[0]
        lm = face_landmarks.landmark

        #Some important landmark 
        cheek_left  = lm[234]
        cheek_right = lm[454]
        nose_left   = lm[49]
        nose_right  = lm[279]
        eye_brow_left  = lm[70]
        eye_brow_right = lm[300]
        chin_left   = lm[172]
        chin_right  = lm[397]
        forehead_mid = lm[10]   
        forehead_left = lm[109]  
        forehead_right = lm[338] 
        cheek_left = lm[234]    
        cheek_right = lm[454]   
        jaw_left = lm[172]       
        jaw_right = lm[397]     
        chin_mid = lm[152]       
        

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
        lower_face_width = calculate_distance(jaw_left, jaw_right)
        middle_face_width = calculate_distance(cheek_left, cheek_right)
        RBot = lower_face_width / middle_face_width if middle_face_width > 0 else 0



        # 6. RTop = Forehead width / Middle face width
        forehead_width = calculate_distance(forehead_left, forehead_right)
        RTop = forehead_width / middle_face_width if middle_face_width > 0 else 0

        

        # 7. RTop - RBot
        RTop_RBot_diff = RTop - RBot

        # 8. fAR = Face width / Face height
        face_width = calculate_distance(cheek_left, cheek_right)
        face_height = calculate_distance(forehead_mid, chin_mid)
        fAR = face_width / face_height if face_height > 0 else 0

        #9. Left cheek width
        left_cheek_distance= calculate_distance(cheek_left, nose_left)

        #10. Right cheek width
        right_cheek_distance= calculate_distance( cheek_right, nose_right)

        #11. Right cheek angle
        right_cheek_angle= calculate_angle(lm[152], lm[400], chin_right)

        #12. Left cheek angle
        left_cheek_angle= calculate_angle(lm[152], lm[176], chin_left)

        #13. Face Length
        face_length = calculate_distance(lm[10], lm[152])

        
        #14. Cheekbone Width
        cheekbone_width = calculate_distance(lm[234], lm[454])

        #15. Jawline Width
        jawline_width = 2 * calculate_distance(lm[454], lm[152])

        #16. Top_jaw Width
        top_jaw_distance = calculate_distance(nose_left, nose_right)

        #17. Forehead Width base on Eyebrow
        forehead_distance = calculate_distance(eye_brow_left, eye_brow_right)

        #18. Chin Width
        chin_distance    = calculate_distance(chin_left, chin_right)
        features = {
            "face_length": face_length,
            "forehead_width": forehead_width,
            "cheekbone_width": cheekbone_width,
            "jawline_width": jawline_width,
            "top_jaw_distance": top_jaw_distance,
            "forehead_distance": forehead_distance,
            "chin_distance": chin_distance,
            "face_rectangularity": face_rectangularity,
            "middle_face_rectangularity": face_middle_rectangularity,
            "forehead_rectangularity": forehead_rectangularity,
            "chin_angle": chin_angle,
            "RBot": RBot,
            "RTop": RTop,
            "RTop_RBot_diff": RTop_RBot_diff,
            "fAR": fAR,
            "left_cheek_distance":left_cheek_distance,
            "right_cheek_distance":right_cheek_distance,
            "right_cheek_angle":right_cheek_angle,
            "left_cheek_angle":left_cheek_angle
        }
        return features

def process_all_images(train_dir):
    
    records = []
    for label in os.listdir(train_dir):
        label_path = os.path.join(train_dir, label)
        if not os.path.isdir(label_path):
            continue
        print(f"Processing label: {label}")
        for filename in os.listdir(label_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(label_path, filename)
                image = cv2.imread(file_path)
                if image is None:
                    print(f"Image can't be read: {file_path}")
                    continue
                features = process_image(image)
                
                if features:
                    record = {"label": label}
                    record.update(features)
                    records.append(record)
                else:
                    print(f"No face found in: {file_path}")
    return records

#Choose you train or test folder HERE
file_type='test'

if file_type =='train':
    file_name='rain'
else:
    file_name='est'


if __name__ == "__main__":
    
    train_directory = fr"FaceShape_Dataset\t{file_name}ing_set"
    records = process_all_images(train_directory)

    
    df = pd.DataFrame(records, columns=["label", "face_length", "forehead_width", "cheekbone_width", "jawline_width",'top_jaw_distance','forehead_distance', 'chin_distance',"face_rectangularity", "middle_face_rectangularity", "forehead_rectangularity", "chin_angle", "RBot",'RTop','RTop_RBot_diff', 'fAR',\
                                 "left_cheek_distance", "right_cheek_distance", "right_cheek_angle", "left_cheek_angle"    ])
    

    # Save as CSV
    csv_filename = f"T{file_name}_data.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Result file is saved as {csv_filename}")
   
 
