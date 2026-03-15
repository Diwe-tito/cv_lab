import cv2
import numpy as np
import matplotlib.pyplot as plt


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
# Load image
# -----------------------------
img = cv2.imread("satellite.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# -----------------------------
# Define 3 ROIs
# -----------------------------
roi1 = gray[200:350, 300:450]   # Region 1
roi2 = gray[400:550, 500:650]   # Region 2
roi3 = gray[100:250, 600:750]   # Region 3 (test region)

# Draw rectangles
cv2.rectangle(img, (300,200), (450,350), (0,255,0), 2)   # green
cv2.rectangle(img, (500,400), (650,550), (255,0,0), 2)   # blue
cv2.rectangle(img, (600,100), (750,250), (0,0,255), 2)   # red


# -----------------------------
# Calculate statistics
# -----------------------------
stats1 = get_stats(roi1)
stats2 = get_stats(roi2)
stats3 = get_stats(roi3)

print("\nRegion 1 Stats:", stats1)
print("Region 2 Stats:", stats2)
print("Region 3 Stats:", stats3)


# -----------------------------
# Compare Region 3 to Region 1 and Region 2
# Using mean, variance, entropy
# -----------------------------
d13 = abs(stats3[0] - stats1[0]) + abs(stats3[1] - stats1[1]) + abs(stats3[6] - stats1[6])
d23 = abs(stats3[0] - stats2[0]) + abs(stats3[1] - stats2[1]) + abs(stats3[6] - stats2[6])

print("\nDistance from Region 3 to Region 1:", d13)
print("Distance from Region 3 to Region 2:", d23)

if d13 < d23:
    print("Region 3 is more similar to Region 1")
else:
    print("Region 3 is more similar to Region 2")


# -----------------------------
# Show image and histograms
# -----------------------------
cv2.imshow("Image with 3 ROIs", img)
cv2.imshow("ROI 1", roi1)
cv2.imshow("ROI 2", roi2)
cv2.imshow("ROI 3", roi3)

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.hist(roi1.ravel(), 256, [0,256])
plt.title("ROI 1 Histogram")

plt.subplot(1,3,2)
plt.hist(roi2.ravel(), 256, [0,256])
plt.title("ROI 2 Histogram")

plt.subplot(1,3,3)
plt.hist(roi3.ravel(), 256, [0,256])
plt.title("ROI 3 Histogram")

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()