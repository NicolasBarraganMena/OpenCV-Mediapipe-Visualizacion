# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 15:08:37 2021

@author: Nicolas Barragan
"""
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

ceja_izq = [70, 63, 105, 66, 107]
ceja_der = [336, 296, 334, 293, 300]
labio_sup = [184, 74, 73, 72, 11, 302, 303, 304, 408]
labio_inf = [77, 90, 180, 85, 16, 315, 404, 320, 307]
parpados_sup = [246, 161, 160, 159, 158, 157, 173, 
                398, 384, 385, 386, 387, 388, 466]
parpados_inf = [7, 163, 144, 145, 153, 154, 155,
                382, 381, 380, 374, 373, 390, 249]

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
            
            '''Ceja izquierda'''
            for index in ceja_izq:
                x = int(face_landmarks.landmark[index].x * width)
                y = int(face_landmarks.landmark[index].y * height)
                cv2.circle(image, (x,y), 2, (255,0,255), 2)
            '''Ceja derecha'''
            for index in ceja_der:
                x = int(face_landmarks.landmark[index].x * width)
                y = int(face_landmarks.landmark[index].y * height)
                cv2.circle(image, (x,y), 2, (255,0,255), 2)
            '''Labio superior'''
            for index in labio_sup:
                x = int(face_landmarks.landmark[index].x * width)
                y = int(face_landmarks.landmark[index].y * height)
                cv2.circle(image, (x,y), 2, (0,255,255), 2)
            '''Labio inferior'''
            for index in labio_inf:
                x = int(face_landmarks.landmark[index].x * width)
                y = int(face_landmarks.landmark[index].y * height)
                cv2.circle(image, (x,y), 2, (0,0,255), 2)
            '''Parpado superior'''
            for index in parpados_sup:
                x = int(face_landmarks.landmark[index].x * width)
                y = int(face_landmarks.landmark[index].y * height)
                cv2.circle(image, (x,y), 2, (255,0,0), 2)
            '''Parpado inferior'''
            for index in parpados_inf:
                x = int(face_landmarks.landmark[index].x * width)
                y = int(face_landmarks.landmark[index].y * height)
                cv2.circle(image, (x,y), 2, (0,255,0), 2)
            
    
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()
