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
    
    image = cv2.imread("Imagen1.jpg")
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image, 
                                      face_landmarks, 
                                      mp_face_mesh.FACEMESH_CONTOURS,
                                      mp_drawing.DrawingSpec(color=(0,255,255), thickness = 1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(255,0,255), thickness = 1))
    
    print("Face landmarks: ", results.multi_face_landmarks)
    
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()
