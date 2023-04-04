import dlib
import cv2
import os, errno
from PIL import Image
from tqdm import tqdm
import natsort

from facePoints import facePoints

import numpy as np
import skimage
from scipy.spatial import ConvexHull

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

INF = 0x3f3f3f

def remove_image(file):
    try:
        os.remove(file)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise

def get_landmark(img):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    allFaces = face_detector(imgRGB, 0)  # detect all faces

    if len(allFaces) != 1:
        return None

    face_rectangle = dlib.rectangle(int(allFaces[0].left()),int(allFaces[0].top()),
                                    int(allFaces[0].right()),int(allFaces[0].bottom()))

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

def cropping(img, RoI, save_path):
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(temp)

    cropped = pil_image.crop(RoI)
    cropped = cropped.resize((112, 112), Image.ANTIALIAS)
    cropped.save(save_path + '.PNG', format='PNG', quality=100)

def make_dir(d):
    path = os.path.join(d)
    os.makedirs(path, exist_ok=True)


def main(path):
    
    if os.path.isdir(path) is False:
        print("[*] Don't exist Raw data directory")
        return
    
    all_ids = os.listdir(path)
    all_ids = natsort.natsorted(all_ids)
    
    for d in tqdm(all_ids):
        cur_dir = os.path.join(path, d)
        
        for img in os.listdir(cur_dir):
            cur_img_path = os.path.join(cur_dir, img)
            
            cur_img = cv2.imread(cur_img_path)
            if cur_img is None:
                remove_image(cur_img_path)
                continue
            
            landmark = get_landmark(cur_img)
            if landmark is None:
                remove_image(cur_img_path)
            

    
if __name__ == '__main__':
    main('K-Celeb/raw')