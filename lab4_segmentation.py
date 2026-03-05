import cv2
import numpy as np

# -----------------------------
# 1. Load Image
# -----------------------------
img = cv2.imread("coins_01_800.jpg")

if img is None:
    print("Image not found")
    exit()

# -----------------------------
# 2. Convert to HSV
# -----------------------------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Detect blue background
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([140, 255, 255])
background_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Invert mask so coins become white
binary = cv2.bitwise_not(background_mask)

# -----------------------------
# 3. Clean Binary Image
# -----------------------------
kernel = np.ones((3,3), np.uint8)

binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

# Smooth edges slightly
binary = cv2.GaussianBlur(binary, (5,5), 0)

# Re-threshold to make sure image is binary again
_, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
# -----------------------------
# get contours
# -----------------------------
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

output = img.copy()
coin_count = 0

output = img.copy()
coin_count = 0

for i, cnt in enumerate(contours):

    area = cv2.contourArea(cnt)

    if area > 3000:

        perimeter = cv2.arcLength(cnt, True)

        circularity = (4 * np.pi * area) / (perimeter ** 2)

        if circularity > 0.75:

            coin_count += 1

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            cv2.drawContours(output, [cnt], -1, (0,255,0), 2)
            cv2.circle(output, (cx, cy), 5, (0,0,255), -1)

            print(f"Coin {coin_count}")
            print(f"  Area: {area}")
            print(f"  Perimeter: {perimeter}")
            print(f"  Circularity: {circularity:.3f}")
            print(f"  Centroid: ({cx}, {cy})")
            print("")

print("Total coins detected:", coin_count)

cv2.imshow("Coin Properties", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

