import ultralytics
from ultralytics import YOLO
import numpy as np
import cv2
import time


model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

state = "idle"
last_time = time.time()
timer=5

player1=""
player2=""
players=[]

game_result = ""

games=["rock","paper","scissors"]

def game_winner(player1="", player2=""):
    if player1 == player2:
        return "Draw!"
    elif player1 == "rock":
        if player2 == "scissors":
            return "Player1 Win!"
        elif player2 == "paper":
            return "Player2 Win!"
    elif player1 == "paper":
        if player2=="scissors":
            return "Player2 Win!"
        elif player2=="rock":
            return "Player1 Win!"
    elif player1=="scissors":
        if player2=="rock":
            return "Player2 Win!"
        elif player2 == "paper":
            return "Player1 Win!"
    else:
        return ""
    

while cap.isOpened():
    ret, frame = cap.read()
    
    
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    
    result = model(frame)[0]   
    if state=="idle":
        cv2.putText(frame, f"{state} - {timer}", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif state=="wait":
        cv2.putText(frame, f"{state} - {timer}", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    if result:       
    
        if len(result.boxes.xyxy) == 2:
            labels = []
            
            for i, xyxy in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = xyxy.numpy().astype('int')
                print(result.boxes.cls)
                print(result.names)
                label = result.names[result.boxes.cls[i].item()].lower()
                labels.append(label)
                players.append("Player "+str(i+1))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(frame, f"{players[i]} {labels[i]}", (x1 + 20, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            player1, player2 = labels[0],labels[1]
            if player1 in games and player2 in games and state == "idle":
                state = "wait"
                last_time = time.time()
            
    if state == "wait":
        timer=int(time.time()-last_time)
    if timer>5:
        timer=5
        state = "result"
        game_result=game_winner(player1,player2)
    if state == "result":
        state="idle"
    cv2.putText(frame, f"{game_result}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("YOLO", frame)

cap.release()
cv2.destroyAllWindows()
