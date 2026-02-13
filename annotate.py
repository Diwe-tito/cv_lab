import cv2
import numpy as np

img = cv2.imread("test.jpg")

img2 = img.copy()

# Text
cv2.putText(img2, "Annotated Image", (50, 400),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (128, 128, 255), 2)

# Line
cv2.line(img2, (0, 0), (300, 300), (255, 0, 255), 2)

# Rectangle outline
cv2.rectangle(img2, (50, 50), (200, 200), (0, 255, 0), 2)

# Filled rectangle
cv2.rectangle(img2, (220, 50), (300, 200), (192, 64, 128), -1)

# Circle outline
cv2.circle(img2, (300, 300), 50, (255, 0, 0), 3)

# Filled circle
cv2.circle(img2, (100, 300), 50, (128, 255, 255), -1)

# Polygon
pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(img2, [pts], True, (0, 0, 255), 2)

cv2.imshow("Annotated Image", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
