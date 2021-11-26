import cv2
import winsound
from time import sleep as s

# Global Vars

box_clr = (0, 0, 255)   # Box Colour  (Blue, Green, Red)
blur_fac = (3, 3)       # Blur Factor (X, Y)
sens = 750              # Sensitivity (lower values give higher sensitivity)
iter = 5                # Iterations  (use lower value for faster processing)
thick = 3               # Box Boundary Thickness
motion = False          # Initialized State

cap = cv2.VideoCapture(0)


def pre_process():
    global blur_fac, frame1, cap, blur

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    diff = cv2.absdiff(frame1, frame2)
    grey = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, blur_fac, 0)


def post_process():
    global blur, iter, contours

    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iter)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


def detect_motion():
    global box_clr, thick, contours, sens, motion

    for c in contours:
        if cv2.contourArea(c) > sens:

            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), box_clr, thick)
            motion = True


def warning():
    global motion

    if motion:
        winsound.PlaySound('alarm.wav', winsound.SND_ASYNC)
        motion = False
        s(0.001)


def display():
    global frame1

    cv2.imshow('Camera', frame1)
    cv2.waitKey(1)


while cap.isOpened():
    pre_process()
    post_process()
    detect_motion()
    warning()
    display()
