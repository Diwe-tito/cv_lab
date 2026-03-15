import cv2
import numpy as np


# -----------------------------
# Function provided by the lab
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
# 1. Load image
# -----------------------------
img = cv2.imread("satellite.png")

if img is None:
    print("Image not found")
    exit()


# -----------------------------
# 2. Convert to grayscale
# -----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# -----------------------------
# 3. Define ROI
# -----------------------------
x1, y1 = 300, 200
x2, y2 = 450, 350

roi = gray[y1:y2, x1:x2]


# -----------------------------
# 4. Draw ROI on original image
# -----------------------------
cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)


# -----------------------------
# 5. Calculate statistics
# -----------------------------
stats = get_stats(roi)

print("Mean:", stats[0])
print("Variance:", stats[1])
print("R:", stats[2])
print("Skewness:", stats[3])
print("Flatness:", stats[4])
print("Uniformity:", stats[5])
print("Entropy:", stats[6])


# -----------------------------
# 6. Display image
# -----------------------------
cv2.imshow("Image with ROI", img)
cv2.imshow("ROI", roi)

cv2.waitKey(0)
cv2.destroyAllWindows()