import cv2
import numpy as np


# -----------------------------
# get_stats function
# -----------------------------
def get_stats(roi, L=256):

    hist = np.histogram(roi, bins=L, range=(0, L), density=True)

    mean = 0
    uniformity = 0
    entropy = 0

    for i in range(L):
        mean += i * hist[0][i]
        uniformity += hist[0][i]**2
        entropy -= hist[0][i] * np.log2(hist[0][i] + 1e-6)

    m = np.zeros(5)

    for n in range(5):
        for zi in range(L):
            m[n] += (zi - mean)**n * hist[0][zi]

    variance = m[2]

    normalised_variance = variance / ((L-1)**2)
    r = 1 - (1 / (1 + normalised_variance))

    skewness = m[3]
    flatness = m[4]

    return mean, variance, r, skewness, flatness, uniformity, entropy


# -----------------------------
# classify function
# -----------------------------
def classify_region(test_stats, ref1_stats, ref2_stats):
    d1 = abs(test_stats[0] - ref1_stats[0]) + \
         abs(test_stats[1] - ref1_stats[1]) + \
         abs(test_stats[6] - ref1_stats[6])

    d2 = abs(test_stats[0] - ref2_stats[0]) + \
         abs(test_stats[1] - ref2_stats[1]) + \
         abs(test_stats[6] - ref2_stats[6])

    if d1 < d2:
        return 1
    else:
        return 2


# -----------------------------
# Load image
# -----------------------------
img = cv2.imread("satellite.png")

if img is None:
    print("Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -----------------------------
# Reference ROIs
# -----------------------------
roi1 = gray[200:350, 300:450]   # texture class 1
roi2 = gray[400:550, 500:650]   # texture class 2

stats1 = get_stats(roi1)
stats2 = get_stats(roi2)

# -----------------------------
# Output image for texture map
# -----------------------------
texture_map = img.copy()

# ROI size and stride
roi_size = 80
stride = 40

# -----------------------------
# Scan across image
# -----------------------------
for y in range(0, gray.shape[0] - roi_size, stride):
    for x in range(0, gray.shape[1] - roi_size, stride):

        roi = gray[y:y+roi_size, x:x+roi_size]
        stats = get_stats(roi)

        label = classify_region(stats, stats1, stats2)

        if label == 1:
            # Region similar to ROI 1 → green
            cv2.rectangle(texture_map, (x, y), (x+roi_size, y+roi_size), (0,255,0), 1)
        else:
            # Region similar to ROI 2 → red
            cv2.rectangle(texture_map, (x, y), (x+roi_size, y+roi_size), (0,0,255), 1)

# Draw original reference ROIs
cv2.rectangle(texture_map, (300,200), (450,350), (0,255,0), 2)
cv2.rectangle(texture_map, (500,400), (650,550), (255,0,0), 2)

# -----------------------------
# Show results
# -----------------------------
cv2.imshow("Texture Map", texture_map)
cv2.imshow("ROI 1", roi1)
cv2.imshow("ROI 2", roi2)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()