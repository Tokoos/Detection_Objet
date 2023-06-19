from ultralytics import YOLO
# from picamera import PiCamera
# import time
from gtts import gTTS
import cv2
import cvzone
import math
import os

cap = cv2.VideoCapture(0) 
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("yolov8n.pt")

classNames = ["personne", "vélo", "voiture", "moto", "avion", "bus", "train", "camion", "bateau",
              "feu de circulation", "fire hydrant", "panneau stop", "parking meter", "banc", "oiseau", "chat",
              "chien", "cheval", "mouton", "vache", "elephant", "ours", "zèbre", "giraffe", "sac à dos", "parapluie",
              "Sac à main", "cravate", "valise", "frisbee", "skis", "snowboard", "sports ball", "cerf-volant", "batte de baseball",
              "gant de baseball", "planche à roulette", "surfboard", "tennis racket", "bouteille", "verre de vin", "tasse",
              "fourchette", "couteau", "cuillère", "bol", "banane", "pomme", "sandwich", "orange", "broccoli",
              "carotte", "hot dog", "pizza", "donut", "gâteau", "chaise", "canapé", "pot de plante", "lit",
              "table à manger", "toilettes", "moniteur de télévision", "ordinateur portable", "souris", "télécommande", "clavier", "telephone",
              "four micro onde", "four", "grille-pain", "sink", "réfrigérateur", "livre", "horloge", "vase", "ciseaux",
              "ours en peluche", "Sèche-cheveux", "brosse à dents", "pen", "peigne"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #booding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            w, h = x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))
            #Confidence  
            conf = math.ceil((box.conf[0]*100))/100
            #Class Name
            cls = int(box.cls[0])
  
            cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0, x1), max(35, y1)))

            language = 'fr'
            myobj = gTTS(text= f'{classNames[cls]}', lang=language, slow=False)
            myobj.save("audio.mp3")
            os.system("mpg321 audio.mp3")

    cv2.imshow("Video", img)
    cv2.waitKey(1)