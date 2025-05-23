from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from pathlib import Path
import cv2
import numpy as np
import time

def distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def angle(a, b, c):
    d=np.arctan2(c[1]-b[1],c[0]-b[0])
    e=np.arctan2(a[1]-b[1],a[0]-b[0])
    angle_=d-e
    if angle_<0:
        angle_=angle_+360
    return np.rad2deg(360-angle_ if angle_>180 else angle_)

def process(image, keypoints):
    
    nose_seen = keypoints[0][0] > 0 and keypoints[0][1] > 0
    left_ear_seen = keypoints[3][0] > 0 and keypoints[3][1] > 0
    right_ear_seen = keypoints[4][0] > 0 and keypoints[4][1] > 0
    
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    
    try:
        
        #if not left_ear_seen and right_ear_seen:
        angle_elbow = angle(right_shoulder, right_elbow, right_wrist)
        x, y = int(right_elbow[0]), int(right_elbow[1])
        #    print("RIGHT")
        #else:
        #    angle_elbow = angle(left_shoulder, left_elbow, left_wrist)
        #    x, y = int(left_elbow[0]), int(left_elbow[1])
        #    print("LEFT")
        
        cv2.putText(image, f"{int(angle_elbow)}", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 25, 255), 2)
        return angle_elbow
    except ZeroDivisionError:
        pass
    return None

model_path = "yolo11n-pose.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture(0)

last_time = time.time()

flag = False
count = 0

writer=cv2.VideoWriter("out.mp4",-1,10,(640,480))

time_not_down=time.time()
while cap.isOpened():
    ret, frame = cap.read()
    cur_time = time.time()
    #writer.write(frame)

    results = model(frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    if not results:
        continue

    result = results[0]
    keypoints = result.keypoints.xy.tolist()
    if not keypoints:
        continue

    keypoints = keypoints[0]
    if not keypoints:
        continue

    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
    annotated = annotator.result()

    angle_ = process(annotated, keypoints)
    
    
    if count > 0 and time.time() - time_not_down > 5:
        count = 0
        time_not_down = time.time()

    if flag and angle_ < 70:  # человек опускается
        time_not_down = time.time()
        count += 1
        flag = False
    elif angle_ > 150:  # человек поднимается
        flag = True

    cv2.putText(frame, f"Count: {count}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (25, 255, 25), 1)
    writer.write(frame)
    cv2.imshow("Pose", annotated)

writer.release()
cap.release()
cv2.destroyAllWindows()
