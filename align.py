import dlib
import cv2
import os
import numpy as np
import math
from PIL import Image
import natsort
from tqdm import tqdm
from datetime import datetime

from facePoints import facePoints

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor(
    'shape_predictor_68_face_landmarks.dat')

INF = 0x3f3f3f

def get_landmark(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    allFaces = face_detector(imgRGB, 0)  # detect all faces

    if len(allFaces) != 1:
        return None

    face_rectangle = dlib.rectangle(int(allFaces[0].left()), int(allFaces[0].top()),
                                    int(allFaces[0].right()), int(allFaces[0].bottom()))

    landmark = landmark_detector(imgRGB, face_rectangle)
    if len(landmark.parts()) != 68:
        return None

    facePoints(img, landmark)

    return landmark


def get_RoI(landmark=None, s=None, e=None):
    x, y, w, h = INF, INF, -INF, -INF

    start = s - 1
    end = e

    for i in range(start, end):
        cur_x = landmark.part(i).x
        cur_y = landmark.part(i).y

        if x > cur_x:
            x = cur_x
        if y > cur_y:
            y = cur_y
        if w < cur_x:
            w = cur_x
        if h < cur_y:
            h = cur_y

    return x, y, w, h

def euclidean_distance(a=None, b=None): 
    x1 = a[0] 
    y1 = a[1] 
    x2 = b[0] 
    y2 = b[1] 
    
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def get_angle(left_eye=None, right_eye=None):
    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2))) 
    left_eye_x = left_eye_center[0] 
    left_eye_y = left_eye_center[1]
    
    right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2))) 
    right_eye_x = right_eye_center[0] 
    right_eye_y = right_eye_center[1]
    
    point_3rd = None
    direction = None
    if left_eye_y < right_eye_y: 
        point_3rd = (right_eye_x, left_eye_y) 
        direction = -1 #rotate same direction to clock 
    else: 
        point_3rd = (left_eye_x, right_eye_y) 
        direction = 1 #rotate inverse direction of clock

    a = euclidean_distance(left_eye_center, point_3rd) 
    b = euclidean_distance(right_eye_center, left_eye_center) 
    c = euclidean_distance(right_eye_center, point_3rd)
    
    cos_a = (b*b + c*c - a*a)/(2*b*c)
    
    angle = np.arccos(cos_a)
    
    angle = (angle * 180) / math.pi
    
    if direction == -1: 
        angle = 90 - angle
        
    return direction, angle

def make_dir(d):
    path = os.path.join(d)
    os.makedirs(path, exist_ok=True)
    
    

def main(path):
    
    if os.path.isdir(path) is False:
        print("[*] Don't exist Raw data directory")
        return
    
    print(f'[*] {datetime.now()} start')
    
    save_path = 'aligned'
    
    all_ids = os.listdir(path)
    all_ids = natsort.natsorted(all_ids)
    
    for d in tqdm(all_ids):
        cur_dir = os.path.join(path, d)
        
        make_dir(os.path.join(save_path, d))
        cnt = 0
        
        for img in os.listdir(cur_dir):
            cur_img_path = os.path.join(cur_dir, img)
            
            cur_img = cv2.imread(cur_img_path)
            if cur_img is None:
                continue
            
            landmark = get_landmark(cur_img)
            if landmark is None:
                continue
            
            left_eye = get_RoI(landmark, 43, 48)
            right_eye = get_RoI(landmark, 37, 42)

            direction, angle = get_angle(left_eye, right_eye)

            origin = cv2.imread(cur_img_path)
            origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
            
            new_img = Image.fromarray(origin) 
            new_img = np.array(new_img.rotate(direction * angle))
            new_img = Image.fromarray(new_img)

            cur_save_path = os.path.join(save_path, d) + '/' + str(cnt) + '.PNG'
            new_img.save(cur_save_path, format='PNG', quality=100)
            
            cnt += 1
    
    print('[*] done')

if __name__ == '__main__':
    main('raw')