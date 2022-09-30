from itertools import count
from pickle import FALSE
import time
from numpy import False_
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

Wr = 0
M0 = 9
M3 = 12

Queue = []
CONF_THRESHOLD = 0.6
Q_NUM =5

Wr = 0
firstF = [1,2,3,4]
secondF = [5,6,7,8]
middleF = [9,10,11,12]
ringF = [13,14,15,16]
pinkyF= [17,18,19,20]

fingerList = [firstF,secondF,middleF,ringF,pinkyF]

def isBent(hand_landmarks,finger):
    bottom = hand_landmarks.landmark[finger[0]].y
    top = hand_landmarks.landmark[finger[3]].y
    if ((top - bottom)>0):
        bent = False
    else:
        bent = True
    return bent


def calVector(finger):
    fvector = [hand_landmarks.landmark[finger[3]].x-hand_landmarks.landmark[Wr].x,
                    hand_landmarks.landmark[finger[3]].y-hand_landmarks.landmark[Wr].y,
                    hand_landmarks.landmark[finger[3]].z-hand_landmarks.landmark[Wr].z]
    return fvector

def ivector(vec):
        x,y,z = vec
        norm = (x**2+y**2+z**2)**(1/2)
        return [x/norm, y/norm, z/norm]

# For webcam input:
cap = cv2.VideoCapture(0)

# Output
isForward = False
UItrigger = 0
# rotateVector
# menuRotate
isselect = False

# local value
isOpen = False # 메뉴 창 열려있는지 아닌지 확인용
trash = 0 # 메뉴에서 넘길 때 몇개 날리는거

with mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        stime = time.time()
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
            continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_height, image_width, _ = image.shape
    # if results.multi_handedness is not None:
    #   for mlh in results.multi_handedness:
    #     if mlh is not None and mlh.classification.label == "Left":
    
    
        if results.multi_hand_landmarks:
        # print(len(results.multi_handedness))
            Queue.append([results.multi_handedness, results.multi_hand_landmarks])
        else:
            Queue.append(None)

        if len(Queue) > Q_NUM:
            q = Queue.pop(0)
            pre_q = q
            if q is not None and len(q)!=0:
                for i, (hand_handedness, hand_landmarks) in enumerate(zip(q[0],q[1])):
                    if len(q[0])==1 and hand_handedness.classification[0].score < CONF_THRESHOLD:
                        if pre_q is not None and len(list(pre_q))==1:
                            pre_mlh, pre_hand_landmarks = pre_q
                            if pre_mlh.classification[0].score >= CONF_THRESHOLD:
                                hand_landmarks = pre_hand_landmarks 
                # 왼손
                    if(hand_handedness.classification[0].label == "Right"):
                        rotationVector = ivector(calVector(fingerList[2]))
                        if(abs(rotationVector[2]/rotationVector[0]) >= 2.14):
                            print("Go Straight")
                            rotationVector = [0.,0.,0.] 
                        else:
                            rotationVector = ivector(calVector(fingerList[2]))
                        #print("rotation ", rotationVector)
                        cv2.putText(image, text="rotation "+str(rotationVector), org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                #오른손
                    if(hand_handedness.classification[0].label == "Left"):
                        fingerBent = []
                        allBent = FALSE
                        for i in range(1,5):
                            currentFinger = fingerList[i]
                            bent = isBent(hand_landmarks, currentFinger)
                            fingerBent.append(bent)
                        # stop sign
                        if(fingerBent.count(True)==4):
                            isForward = False
                            UItrigger = 0
                            # print("stop!")
                            cv2.putText(image, text="stop", org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        # go sign
                        elif(fingerBent.count(False) == 4):
                            if(isOpen == True):
                                isOpen = False
                                UItrigger = 2
                                cv2.putText(image, text="Close",org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=2)
                            else:
                                isForward= True
                                UItrigger = 0
                                # print("Go!")
                                cv2.putText(image, text="Go",org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=2)
                    # UI open sign
                    # elif(fingerBent[0]==False and fingerBent[1:].count(True) == 3 and isOpen == False ):
                        elif(fingerBent[0] == True and fingerBent.count(False) == 3):
                            if(isOpen == False):
                                isOpen = True
                                UItrigger = 1
                                print("open")
                                # print("UI open trigger")
                                cv2.putText(image, text="UI", org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                            else:
                                UItrigger = 0
                                second = ivector(calVector(fingerList[1]))
                                if(isselect == True):
                                    isselect = False      
                                # print(second)
                                if(trash == 0): 
                                    if(second[2]<-0.35):
                                        isselect = True
                                        print("select")
                                        cv2.putText(image, text="right", org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                                    if(second[0]<-0.3):
                                        menuRotate = 1
                                        print("right")
                                        cv2.putText(image, text="right", org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                                    elif(second[0]>0.3):
                                        menuRotate = -1
                                        print("left")
                                        cv2.putText(image, text="left", org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                                if(trash <= 5):
                                    trash += 1
                                elif(trash >5):
                                    trash = 0
            
                # print(fingerBent)
                # UI close sign
                # elif(fingerBent[0]==False and fingerBent[1:].count(True) == 3 and UItrigger == True):
                #     isOpen = False
                #     UItrigger = 2
                #     # print("UI open close")
                #     cv2.putText(image, text="UI open close", org=(60,60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

        etime = time.time()
        # print(etime-stime)
        # Flip the image horizontally for a selfie-view display.
        # print(isForward)
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()