import numpy as np
import cv2

# Create black image (500x500, 3 channels)
img = np.zeros((500, 500, 3), dtype=np.uint8)
img2= np.ones((1000, 1000, 3), dtype=np.uint8) * 255

# Yellow
cv2.circle(img2, (375, 500), 100, (0, 200, 250), 20)

# Green
cv2.circle(img2, (625, 500), 100, (0, 125, 0), 20)

# Blue
cv2.circle(img2, (250, 400), 100, (255, 0, 0), 20)

# Black
cv2.circle(img2, (500, 400), 100, (0, 0, 0), 20)

# Red
cv2.circle(img2, (750, 400), 100, (0, 0, 255), 20)



##cv2.imshow("Black Image", img)
cv2.imshow("OLympic Rings ", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
 