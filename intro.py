import cv2
import numpy as np

image = cv2.imread("img/food.jpeg")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

low_biru = np.array([100, 50, 50])
up_biru = np.array([140, 255, 255])

mask = cv2.inRange(hsv, low_biru, up_biru)

filtered = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow("img", image)
cv2.imshow("filtered", filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()