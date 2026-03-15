import cv2
import numpy as np

# -----------------------------
# 1. Load image
# -----------------------------
img = cv2.imread("coins_02_800.jpg")

if img is None:
    print("Image not found")
    exit()

output = img.copy()

# -----------------------------
# 2. Convert to grayscale
# -----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 3. Otsu threshold
# -----------------------------
threshold, img_thresh = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# -----------------------------
# 4. Morphological operations
# -----------------------------
strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# remove noise
img_eroded = cv2.erode(img_thresh, strel, iterations=1)

# restore shape
img_clean = cv2.dilate(img_eroded, strel, iterations=2)

# -----------------------------
# 5. Find contours
# -----------------------------
contours, hierarchy = cv2.findContours(
    img_clean,
    cv2.RETR_CCOMP,
    cv2.CHAIN_APPROX_SIMPLE
)

# -----------------------------
# 6. Count coins
# -----------------------------
coin_count = 0

fcoin_count = 0

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area > 800:

        coin_count += 1

        # centroid
        m = cv2.moments(cnt)
        cx = int(m["m10"]/m["m00"])
        cy = int(m["m01"]/m["m00"])

        # classify by area
        if area < 2000:
            coin_type = "5p / small"
        elif area < 3500:
            coin_type = "10p"
        elif area < 5000:
            coin_type = "20p / 50p"
        else:
            coin_type = "£1 / large"

        # draw contour
        cv2.drawContours(output,[cnt],-1,(0,255,0),2)

        # draw label
        cv2.putText(
            output,
            coin_type,
            (cx-30, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,255),
            1
        )

print("Coins detected:", coin_count)

# -----------------------------
# 7. Display images
# -----------------------------
cv2.imshow("Original", img)
cv2.imshow("Threshold", img_thresh)
cv2.imshow("Clean Mask", img_clean)
cv2.imshow("Detected Coins", output)

# press q to close
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()