# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:08:37 2021

@author: Nicolas Barragan
"""
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

with mp_face_mesh.FaceMesh(
        static_image_mode = True,
        max_num_faces = 1,
        min_detection_confidence = 0.5) as face_mesh:
    
    image = cv2.imread("leonmessibarca111.jpg")
    cv2.imshow("Image", image)
    cv2.waitKey(0)
cv2.destroyAllWindows()
