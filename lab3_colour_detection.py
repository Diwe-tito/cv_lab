import cv2
import numpy as np

# Load image
img = cv2.imread("test.jpg")

if img is None:
    print("Image not found")
    exit()

# Convert to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# ----------------------------
# Choose colour range (example: GREEN)
# ----------------------------

lower = np.array([0, 100, 100])
upper = np.array([10, 255, 255])

# Create mask
mask = cv2.inRange(img_hsv, lower, upper)

# Extract colour using mask
result = cv2.bitwise_and(img, img, mask=mask)

# ----------------------------
# Optional: Find contours
# ----------------------------

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:  # ignore tiny noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

# ----------------------------
# Show results
# ----------------------------

cv2.imshow("Original", img)
cv2.imshow("Mask", mask)
cv2.imshow("Detected Colour", result)

cv2.waitKey(0)
cv2.destroyAllWindows()