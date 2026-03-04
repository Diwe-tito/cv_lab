import cv2
import numpy as np

# -----------------------------
# 1. Load Image
# -----------------------------
img = cv2.imread("coins_02_800.jpg")

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

# -----------------------------
# 4. Watershed Segmentation
# -----------------------------

# Sure background
sure_bg = cv2.dilate(binary, kernel, iterations=3)

# Distance transform (find coin centers)
dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

# Threshold for sure foreground (0.20 worked best for you)
_, sure_fg = cv2.threshold(
    dist_transform,
    0.20 * dist_transform.max(),
    255,
    0
)

sure_fg = np.uint8(sure_fg)

# Unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
_, markers = cv2.connectedComponents(sure_fg)

markers = markers + 1
markers[unknown == 255] = 0

# Apply watershed
markers = cv2.watershed(img, markers)

# -----------------------------
# 5. Extract and Count Coins
# -----------------------------
output = img.copy()
coin_count = 0

for label in np.unique(markers):

    # Skip background and boundary
    if label <= 1:
        continue

    mask = np.zeros(binary.shape, dtype="uint8")
    mask[markers == label] = 255

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter very small regions
        if area > 3000:
            coin_count += 1
            cv2.drawContours(output, [cnt], -1, (0,255,0), 2)

print("Number of coins detected:", coin_count)

# -----------------------------
# 6. Show Results
# -----------------------------
cv2.imshow("Binary Mask", binary)
cv2.imshow("Segmented Coins", output)

cv2.waitKey(0)
cv2.destroyAllWindows()