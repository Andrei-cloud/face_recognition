#! /usr/bin/env python
import cv2
import numpy as np
import math
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from PIL import Image
import matplotlib.pyplot as plt
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine

path = 'knownfaces'
faceCascade = cv2.CascadeClassifier('haar_cascade.xml')
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
profileCascade = cv2.CascadeClassifier("haarcascade_profile.xml")
images = []
names = []
myList = os.listdir(path)
tolerance = 0.4
every = 12
width = 800
height = 600

for cl in myList:
    if os.path.splitext(cl)[1].lower() == '.jpg':
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        names.append(os.path.splitext(cl)[0])

def face_detection(img):
    facesLoc=[]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    facesLoc = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    return facesLoc

def im_align(img):
    #print(type(img))
    gi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eyeCascade.detectMultiScale(gi, minNeighbors=3)
    if len(img) == 0:
        return img
    #print(eyes)
    eyes = sorted(eyes, key=lambda x: x[2], reverse=True)[:2] # sort by width and take two widest
    #print(eyes)
    index = 0
    
    for (eye_x, eye_y, eye_w, eye_h) in eyes:
       if index == 0:
          eye_1 = (eye_x, eye_y, eye_w, eye_h)
       elif index == 1:
          eye_2 = (eye_x, eye_y, eye_w, eye_h)
#        eye_border = Rectangle((eye_x, eye_y), eye_w, eye_h, fill=False, color='green')
#        ax.add_patch(eye_border)
       index = index + 1
    
    if eye_1[0] < eye_2[0]:
       left_eye = eye_1
       right_eye = eye_2
    else:
       left_eye = eye_2
       right_eye = eye_1
    
    left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
    left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]

    right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
    right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
    
    if left_eye_y < right_eye_y:
       point_3rd = (right_eye_x, left_eye_y)
       direction = 1 #rotate same direction to clock
       print("rotate to clock direction")
    else:
       point_3rd = (left_eye_x, right_eye_y)
       direction = -1 #rotate inverse direction of clock
       print("rotate to inverse clock direction")
    
    a = euclidean_distance(left_eye_center, point_3rd)
    b = euclidean_distance(right_eye_center, left_eye_center)
    c = euclidean_distance(right_eye_center, point_3rd)

    cos_a = (b*b + c*c - a*a)/(2*b*c)
    print("cos(a) = ", cos_a)

    angle = np.arccos(cos_a)
    print("angle: ", angle," in radian")

    angle = (angle * 180) / math.pi
    print("angle: ", angle," in degree")
     
    if direction == -1:
       angle = 90 - angle

    new_img = Image.fromarray(img)
    if angle < 45:
        new_img = np.array(new_img.rotate(direction * angle))
    
    return np.asarray(new_img)
    
def extract_face_from_image(images, faces=None, required_size=(224, 224)):
    face_images = []  
    for img in images:
        if faces is None:
            faces = face_detection(img)
        for face in faces:
            # extract the bounding box from the requested face
            x1, y1, width, height = face
            x2, y2 = x1 + width, y1 + height

            face_boundary = img[y1:y2, x1:x2]

            # resize pixels to the model size
            face_image = Image.fromarray(face_boundary)
            face_image = face_image.resize(required_size)
            face_array = np.asarray(face_image)
            #face_array = face_array / 255
            #face_array = im_align(face_array)
            if face_array is not None:
                face_images.append(face_array)
    return face_images

def create_model():
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224,224,3), pooling='avg')
    return model    

def get_model_scores(faces, model):
    samples = np.asarray(faces, 'float32')
    # prepare the data for the model
    samples = preprocess_input(samples, version=2)
    # perform prediction
    return model.predict(samples)

def im_show(img, name, time):
    cv2.namedWindow(name)
    cv2.moveWindow(name, 900,-900)
    cv2.imshow(name, img)
    cv2.waitKey(time)
    return

def area(loc):
    return loc[2]*loc[3]

def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))



model = create_model()
model_scores = get_model_scores(extract_face_from_image(images), model)
print('Encoding complete')
#print(model_scores)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cycle = 1
while True:
    success, img = cap.read()
    img = cv2.flip(img, +1)
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25, )
    facelocs = face_detection(img)
    if cycle == every:
        cycle = 0
    else:
        cycle += 1
    name = ''

    for face in facelocs:
        #print(face)
        is_match= []
        if len(face) >0:
            x,y,w,h = face

            #img = im_align(img)
            extracted = extract_face_from_image([img], [face])
            #im_show(extracted[0], 'Face', 1)
            score = get_model_scores(extracted, model)
            distances = [cosine(score, x) for x in model_scores]
            is_match = [x <= tolerance for x in distances]
            #print(is_match)
            match = np.argmin(distances)
            name = names[match].upper()
        
            x2, y2 = x+w, y+h
            if is_match[match]:
                cv2.rectangle(img, (x-5,y-5), (x2+5,y2+5), (0,200,0), 2)
                cv2.rectangle(img, (x-5,y2),(x2+5,y2+35), (0,200,0), cv2.FILLED)
                cv2.putText(img, name, (x+6,y2+28), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
            else:
                cv2.rectangle(img, (x-5,y-5), (x2+5,y2+5), (0,0,255), 2)
                cv2.rectangle(img, (x-5,y2),(x2+5,y2+35), (0,0,255), cv2.FILLED)
                cv2.putText(img, 'UNKNOWN', (x+6,y2+28), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)

    cv2.imshow('Webcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # Press '''q''' to exit
        break