import cv2
import numpy as np
import time
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer/trainingdata.yml')
id=0
cam = cv2.VideoCapture(0)
import pyrebase
#import urllib.request
#url="http://192.168.1.93.8080"
config = {
  "apiKey": "AIzaSyB49YssOwHufdBtIZOtDgtxtfxDvYLKbRs",
  "authDomain": "python-30649.firebaseapp.com",
  "databaseURL": "https://python-30649-default-rtdb.firebaseio.com",
  "storageBucket": "python-30649.appspot.com"}
firebase = pyrebase.initialize_app(config)
# Get a reference to the auth service
auth = firebase.auth()
# Log the user in
user = auth.sign_in_with_email_and_password("yasminbensaad54@gmail.com", "yasminebs")
# Get a reference to the database service
db = firebase.database()
while(True):
    #imageResponse = urllib.request.urlopen(url)
    #img=np.array[bytearry(imgResponse.read()),dtype == np.uint8]
   # img = cv2.imdecode(img,-1)
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])
        if conf<=70:
            if id ==1:
                id='yas'
            elif id ==2:
                id=='smida'
            elif id ==3:
                id=='malek'
        else:
            id='inconnu'
       # data to save
        data = { "name": id,"time":time.ctime(),"confidence":conf}
       # Pass the user's idToken to the push method
        results = db.child("facerec").child(id).set(data)
        cv2.putText(img,str(id),(x,y+2),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
        cv2.putText(img,str(conf),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
    cv2.imshow('reconnaissance faciale',img)

    if(cv2.waitKey(1)==27):
        break
cam.release
cv2.destroyAllWindows()
