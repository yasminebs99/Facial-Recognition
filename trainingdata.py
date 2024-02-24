import numpy as np
import cv2
import PIL
from PIL import Image
import os
recognizer=cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'
def getimageswithid(path):
    imagepaths= [os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    for imagepath in imagepaths:
        faceimag= Image.open(imagepath).convert('L')
        facenp=np.array(faceimag,'uint8')
        id=int(os.path.split(imagepath)[-1].split('.')[1])
        faces.append(facenp)
        print(id)
        ids.append(id)
        cv2.imshow('training',facenp)
        cv2.waitKey(10)
    return ids,faces
ids,faces=getimageswithid(path)
recognizer.train(faces,np.array(ids))
recognizer.save('recognizer/trainingdata.yml')
cv2.destroyAllWindows()
