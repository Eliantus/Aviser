import cv2
import numpy as np
from keras.models import load_model
import pyttsx3
from pygame import mixer
import random
import tkinter as tk
from threading import Thread

#Model Importation
model=load_model("./model2-008.model")
mixer.init()
sound = mixer.Sound('beep1.wav')

#Voice initialization
converter = pyttsx3.init()
converter.setProperty('rate', 180)
converter.setProperty('volume', 1)
voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_frFR_PaulM"
converter.setProperty('voice', voice_id)
converter.runAndWait()

#Load the text
with open("text.txt",encoding='UTF-8') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

#Access Denied function
def refus():
    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.configure(background='red')
    tk.Label(root, text='Accès refusé', font=("Arial", 35)).place(relx=.5, rely=.5, anchor="c")
    root.after(1500, lambda: root.destroy())     # time in ms
    root.mainloop()

#Access Granted function
def acces():
    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.configure(background='green')
    tk.Label(root, text='Accès autorisé', font=("Arial", 35)).place(relx=.5, rely=.5, anchor="c")
    root.after(3000, lambda: root.destroy())     # time in ms
    root.mainloop()

#Warning talk function
def aviser ():
     p=random.randint(0, len(lines)-1)
     converter.say(lines[p])
     converter.runAndWait()
     
     
#Initialize label     
labels_dict={0:'No Mask',1:'Mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

#Choose camera
size = 4
webcam = cv2.VideoCapture(1)

# We load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

i=0
j=0
k=0

while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,1) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)     
    
    if(len(faces)==0):
        i=j=0
    
    # Draw rectangles around each face
    for f in faces:
                
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        #Save just the rectangle faces in SubRecFaces
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(150,150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)
        
        label=np.argmax(result,axis=1)[0]
        
        """if(label == 1):
            j=j+1
            i=0
            print("j = ",j)
            
        if(j!=0 and j%3==0):    
            print("j = ",j)
            Thread(target = acces).start()"""
        
        if(label == 0):
            sound.play()
            Thread(target = refus).start()
            i=i+1
            j=0
            print("i = ",i)
            
        if(i==5 or (i!=0 and i%20==0)):    
            print("i = ",i)
            Thread(target = aviser).start()
            cv2.imwrite("./guilty/"+"no_mask_"+str(k)+".jpg", im)
            k=k+1
           
        
        cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    # Show the image
    cv2.imshow('AVISER - Mask detector',   im)
    
    
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()