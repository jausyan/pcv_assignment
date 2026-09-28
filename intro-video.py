import cv2
import numpy as np

cap = cv2.VideoCapture("img/flight1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_biru = np.array([100, 50, 50])
    up_biru = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, low_biru, up_biru)
    filtered = cv2.bitwise_and(frame, frame, mask=mask)


    cv2.imshow("Real Video", frame)
    cv2.imshow("Filtered Video Window", filtered)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()