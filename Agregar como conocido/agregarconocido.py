# -*- coding: utf-8 -*-
"""
Created on Sat Nov 07 10:59:06 2021

@author: Nicolas Barragan
"""

from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import mediapipe as mp
import numpy as np

def iniciar():
    global cap
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    visualizar()
    
def visualizar():
    global cap
    if cap is not None:
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
            ret, frame = cap.read()
            if ret == True:
                
                color_peligro = (0,0,0)
                salida = 'Agregar como conocido?'
                
                if selected.get() == 1:
                    color_peligro = (0,255,0)
                    salida = "Agregado como amigo"
                if selected.get() == 2:
                    color_peligro = (255,0,0)
                    salida = "Agregado como no grato"
                if selected.get() == 3:
                    color_peligro = (0,0,0)
                    salida = "Agregar como conocido?"
                
                frame = imutils.resize(frame, width=640)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = frame.shape
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = selfie_segmentation.process(frame_rgb)
        
                bg_image = np.ones(frame.shape, dtype = np.uint8)
                bg_image[:] = color_peligro
                
                _, th = cv2.threshold(results.segmentation_mask, 0.75, 255, cv2.THRESH_BINARY)
                th = th.astype(np.uint8)
        
                bg = cv2.bitwise_and(bg_image, bg_image,mask = th) 
                silueta = cv2.bitwise_and(frame, frame, mask = th)
                output = cv2.add(bg, frame)
        
                cv2.putText(output, salida, (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_peligro, 2)
        
                im = Image.fromarray(output)
                img = ImageTk.PhotoImage(image=im)
    
                lblVideo.configure(image=img)
                lblVideo.image = img
                lblVideo.after(10, visualizar)
            else:
                lblVideo.image = ""
                cap.release()

def finalizar():
    global cap
    cap.release()

mp_selfie_segmentation = mp.solutions.selfie_segmentation    
cap = None
root = Tk()

btnIniciar = Button(root, text="Iniciar", width=45, command=iniciar)
btnIniciar.grid(column=0, row=0, padx=5, pady=5)

btnFinalizar = Button(root, text="Finalizar", width=45, command=finalizar)
btnFinalizar.grid(column=1, row=0, padx=5, pady=5)

# Label ¿Como lo quieres agregar?
lblInfo2 = Label(root, text="¿Como lo quieres agregar?", width=25)
lblInfo2.grid(column=0, row=1, padx=5, pady=5)
selected = IntVar()
rad1 = Radiobutton(root, text='Amigo', width=25,value=1, variable=selected)#, command= deteccion_color)
rad2 = Radiobutton(root, text='No grato',width=25, value=2, variable=selected)#, command= deteccion_color)
rad3 = Radiobutton(root, text='Reiniciar',width=25, value=3, variable=selected)#, command= deteccion_color)
rad1.grid(column=0, row=2)
rad2.grid(column=1, row=2)
rad3.grid(column=0, row=3)


lblVideo = Label(root)
lblVideo.grid(column=0, row=4, columnspan=2)

root.mainloop()