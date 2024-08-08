from django.shortcuts import render, redirect
from .forms import usernameForm, DateForm, UsernameAndDateForm, DateForm_2
from django.contrib import messages
from django.contrib.auth.models import User
import cv2
import dlib
import imutils
from imutils.video import VideoStream
from imutils.face_utils import rect_to_bb, FaceAligner
import time
from attendance_system_facial_recognition.settings import BASE_DIR
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from django.contrib.auth.decorators import login_required
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
from django_pandas.io import read_frame
from users.models import Present, Time
import seaborn as sns
import pandas as pd
from django.db.models import Count
#import mpld3
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
import math

mpl.use('Agg')


#utility functions:
def username_present(username):
    if User.objects.filter(username=username).exists():
        return True

    return False

def create_dataset(username):
    id = username
    if not os.path.exists('face_recognition_data/training_dataset/{}/'.format(id)):
        os.makedirs('face_recognition_data/training_dataset/{}/'.format(id))
    directory = 'face_recognition_data/training_dataset/{}/'.format(id)

    # Detect face
    # Loading the HOG face detector and the shape predictpr for allignment

    print("[INFO] Loading the facial detector")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   # Add path to the shape predictor
    fa = FaceAligner(predictor, desiredFaceWidth=96)
    # capture images from the webcam and process and detect the face
    # Initialize the video stream
    print("[INFO] Initializing Video stream")
    vs = VideoStream(src=0).start()
    # time.sleep(2.0) ####CHECK######

    # Our identifier
    # We will put the id here and we will store the id with a face, so that later we can identify whose face it is

    # Our dataset naming counter
    sampleNum = 0
    # Capturing the faces one by one and detect the faces and showing it on the window
    while(True):
        # Capturing the image
        # vs.read each frame
        frame = vs.read()
        # Resize each image
        frame = imutils.resize(frame, width=800)
        # the returned img is a colored image but for the classifier to work we need a greyscale image
        # to convert
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # To store the faces
        # This will detect all the images in the current frame, and it will return the coordinates of the faces
        # Takes in image and some other parameter for accurate result
        faces = detector(gray_frame, 0)
        # In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.

        for face in faces:
            print("inside for loop")
            (x, y, w, h) = face_utils.rect_to_bb(face)

            face_aligned = fa.align(frame, gray_frame, face)
            # Whenever the program captures the face, we will write that is a folder
            # Before capturing the face, we need to tell the script whose face it is
            # For that we will need an identifier, here we call it id
            # So now we captured a face, we need to write it in a file
            sampleNum = sampleNum + 1
            # Saving the image dataset, but only the face part, cropping the rest

            if face is None:
                print("face is none")
                continue

            cv2.imwrite(directory + '/' + str(sampleNum) + '.jpg', face_aligned)
            face_aligned = imutils.resize(face_aligned, width=400)
            # cv2.imshow("Image Captured", face_aligned)
            # @params the initial point of the rectangle will be x, y and
            # @params end point will be x+width and y+height
            # @params along with color of the rectangle
            # @params thickness of the rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # Before continuing to the next loop, I want to give it a little pause
            # waitKey of 100 millisecond
            cv2.waitKey(50)

        # Showing the image in another window
        # Creates a window with window name "Face" and with the image img
        cv2.imshow("Add Images", frame)
        # Before closing it we need to give a wait command, otherwise the open cv wont work
        # @params with the millisecond of delay 1
        cv2.waitKey(1)
        # To get out of the loop
        if(sampleNum > 300):
            break

    # Stopping the video stream
    vs.stop()
    # destroying all the windows
    cv2.destroyAllWindows()


def predict(face_aligned, svc, threshold=0.7):
    face_encodings = np.zeros((1, 128))
    try:
        x_face_locations = face_recognition.face_locations(face_aligned)
        faces_encodings = face_recognition.face_encodings(face_aligned, known_face_locations=x_face_locations)
        if(len(faces_encodings) == 0):
            return ([-1], [0])

    except:
        return ([-1], [0])

    prob = svc.predict_proba(faces_encodings)
    result = np.where(prob[0] == np.amax(prob[0]))
    if(prob[0][result[0]] <= threshold):
        return ([-1], prob[0][result[0]])

    return (result[0], prob[0][result[0]])


def vizualize_Data(embedded, targets):

    X_embedded = TSNE(n_components=2).fit_transform(embedded)

    for i, t in enumerate(set(targets)):
        idx = targets == t
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

    plt.legend(bbox_to_anchor=(1, 1))
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    plt.savefig('./recognition/static/recognition/img/training_visualisation.png')
    plt.close()


def update_attendance_in_db_in(present):
    today = datetime.date.today()
    time = datetime.datetime.now()
    for person in present:
        user = User.objects.get(username=person)
        try:
            qs = Present.objects.get(user=user, date=today)
        except:
            qs = None

        if qs is None:
            if present[person] == True:
                a = Present(user=user, date=today, present=True)
                a.save()
            else:
                a = Present(user=user, date=today, present=False)
                a.save()
        else:
            if present[person] == True:
                qs.present = True
                qs.save(update_fields=['present'])
        if present[person] == True:
            a = Time(user=user, date=today, time=time, out=False)
            a.save()


def update_attendance_in_db_out(present):
    today = datetime.date.today()
    time = datetime.datetime.now()
    for person in present:
        user = User.objects.get(username=person)
        if present[person] == True:
            a = Time(user=user, date=today, time=time, out=True)
            a.save()


def check_validity_times(times_all):

    if(len(times_all) > 0):
        sign = times_all.first().out
    else:
        sign = True
    times_in = times_all.filter(out=False)
    times_out = times_all.filter(out=True)
    if(len(times_in) != len(times_out)):
        sign = True
        break_hourss = 0
        if(sign == True):
            check = False
            break_hourss = 0
            return (check, break_hourss)
    prev = True
    prev_time = times_all.first().time

    for obj in times_all:
        curr = obj.out
        if(curr == prev):
            check = False
            break_hourss = 0
            return (check, break_hourss)
        if(curr == False):
            curr_time = obj.time
            to = curr_time
            ti = prev_time
            diff = to - ti
            break_seconds = diff.total_seconds()
            break_hours = break_seconds/3600
            break_hourss += break_hours
        prev_time = obj.time
        prev = curr

    check = True
    return (check, break_hourss)


def mark_your_attendance():
    global attendance
    present = {}
    today = datetime.date.today()
    time = datetime.datetime.now()
    myDict = {}
    print("[INFO] loading encodings + face detector...")
    data = pickle.loads(open("face_recognition_data/encodings.pickle", "rb").read())
    detector = cv2.CascadeClassifier("face_recognition_data/haarcascade_frontalface_default.xml")
    embedder = cv2.dnn.readNetFromTorch("face_recognition_data/openface_nn4.small2.v1.t7")
    recognizer = pickle.loads(open("face_recognition_data/recognizer.pickle", "rb").read())
    le = pickle.loads(open("face_recognition_data/le.pickle", "rb").read())
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    start = {}
    count = {}
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 20 or fH < 20:
                    continue
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = le.classes_[j]
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                if (name != 'unknown'):
                    if (name not in start):
                        start[name] = time.time()
                        count[name] = 0
                    if count[name] >= 4 and (time.time() - start[name]) > 1.2:
                        myDict[name] = 1
                        attendance[name] = 'P'
                        present[name] = True
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      (0, 255, 0), 2)
                        cv2.putText(frame, name, (startX, startY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    elif count[name] < 4:
                        count[name] += 1
                    else:
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      (0, 255, 0), 2)
                        cv2.putText(frame, name, (startX, startY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, name, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        else:
            update_attendance_in_db_in(present)
            print("Attendance marked for: ", present)

    cv2.destroyAllWindows()
    vs.stop()

