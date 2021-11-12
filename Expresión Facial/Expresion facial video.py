# -*- coding: utf-8 -*-
"""
Created on Thu Nov 05 14:19:54 2021

@author: Nicolas Barragan
"""

import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
'''
ceja_izq = [70, 63, 105, 66, 107]
ceja_der = [336, 296, 334, 293, 300]
labio_sup = [184, 74, 73, 72, 11, 302, 303, 304, 408]
labio_inf = [77, 90, 180, 85, 16, 315, 404, 320, 307]
parpados_sup = [246, 161, 160, 159, 158, 157, 173, 
                398, 384, 385, 386, 387, 388, 466]
parpados_inf = [7, 163, 144, 145, 153, 154, 155,
                382, 381, 380, 374, 373, 390, 249]
'''
ceja_izq = 65
ojo_izq = 158
ceja_der = 295
ojo_der = 385
extremo_der = 308
extremo_izq = 78
labio_inf = 14
labio_sup = 13

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_mesh.FaceMesh(
        static_image_mode = False,
        max_num_faces = 1,
        min_detection_confidence = 0.5) as face_mesh:
    
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        
        frame = cv2.flip(frame,1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                
                '''Ceja izquierda - ojo izquierdo'''
                p1 = [int(face_landmarks.landmark[ceja_izq].x * width), 
                      int(face_landmarks.landmark[ceja_izq].y * height)]
                cv2.circle(frame, p1, 2, (255,0,255), 2)
                p2 = [int(face_landmarks.landmark[ojo_izq].x * width), 
                      int(face_landmarks.landmark[ojo_izq].y * height)]
                cv2.circle(frame, p2, 2, (255,0,255), 2)
                d1 = str(int(math.dist(p1, p2)))
                cv2.putText(frame, d1, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
                '''Ceja derecha - ojo derecho'''
                p3 = [int(face_landmarks.landmark[ceja_der].x * width), 
                      int(face_landmarks.landmark[ceja_der].y * height)]
                cv2.circle(frame, p3, 2, (0,255,255), 2)
                p4 = [int(face_landmarks.landmark[ojo_der].x * width), 
                      int(face_landmarks.landmark[ojo_der].y * height)]
                cv2.circle(frame, p4, 2, (0,255,255), 2)
                d2 = str(int(math.dist(p3, p4)))
                cv2.putText(frame, d2, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                '''Extremos boca'''
                p5 = [int(face_landmarks.landmark[extremo_der].x * width), 
                      int(face_landmarks.landmark[extremo_der].y * height)]
                cv2.circle(frame, p5, 2, (0,0,255), 2)
                p6 = [int(face_landmarks.landmark[extremo_izq].x * width), 
                      int(face_landmarks.landmark[extremo_izq].y * height)]
                cv2.circle(frame, p6, 2, (0,0,255), 2)
                d3 = str(int(math.dist(p5, p6)))
                cv2.putText(frame, d3, (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                '''Apertura boca'''
                p7 = [int(face_landmarks.landmark[labio_inf].x * width), 
                      int(face_landmarks.landmark[labio_inf].y * height)]
                cv2.circle(frame, p7, 2, (255,0,0), 2)
                p8 = [int(face_landmarks.landmark[labio_sup].x * width), 
                      int(face_landmarks.landmark[labio_sup].y * height)]
                cv2.circle(frame, p8, 2, (0,255,0), 2)
                d4 = str(int(math.dist(p7, p8)))
                cv2.putText(frame, d4, (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                
        
        cv2.imshow("Video", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
        