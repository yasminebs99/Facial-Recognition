import numpy as np
import cv2
id= input("donner user id :")
faceDetect= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
SampleNum=0
while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,h,w) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        SampleNum=SampleNum+1
        cv2.imwrite("dataset/user."+str(id)+'.'+str(SampleNum)+".png",gray[y:y+h,x:x+w])
    cv2.imshow('reconnaissance faciale',img)
    cv2.waitKey(100)
    if (SampleNum>30):
        break
cam.release()
cv2.destroyAllWindows()
