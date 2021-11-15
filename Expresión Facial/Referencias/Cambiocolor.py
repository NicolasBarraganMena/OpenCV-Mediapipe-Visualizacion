# -*- coding: utf-8 -*-
"""
Created on Fri Nov 06 14:03:02 2021

@author: Nicolas
"""

import cv2
import mediapipe as mp
import numpy as np

mp_selfie_segmentation = mp.solutions.selfie_segmentation

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

color_peligro = (0,0,255)

with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=0) as selfie_segmentation:
    
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(frame_rgb)
        
        bg_image = np.ones(frame.shape, dtype = np.uint8)
        bg_image[:] = color_peligro
        
        _, th = cv2.threshold(results.segmentation_mask, 0.75, 255, cv2.THRESH_BINARY)
        th = th.astype(np.uint8)
        
        bg = cv2.bitwise_and(bg_image, bg_image,mask = th)
        
        silueta = cv2.bitwise_and(frame, frame, mask = th)
        
        output = cv2.add(bg, frame)
               
        #cv2.imshow("mascara de segmentacion", results.segmentation_mask)
        #cv2.imshow("Color peligro", bg_image)
        #cv2.imshow("umbral", th)
        cv2.imshow("Silueta", silueta)
        cv2.imshow("Color silueta", bg)
        cv2.imshow("Output", output)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF==27:
            break

cap.release()
cv2.destroyAllWindows()