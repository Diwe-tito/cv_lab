import cv2
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# get_stats function (same as before)
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
# Define TWO ROIs
# -----------------------------
roi1 = gray[200:350, 300:450]   # example: forest
roi2 = gray[400:550, 500:650]   # example: field


# Draw rectangles
cv2.rectangle(img,(300,200),(450,350),(0,255,0),2)
cv2.rectangle(img,(500,400),(650,550),(255,0,0),2)


# -----------------------------
# Calculate statistics
# -----------------------------
stats1 = get_stats(roi1)
stats2 = get_stats(roi2)

print("\nRegion 1 Stats")
print(stats1)

print("\nRegion 2 Stats")
print(stats2)


# -----------------------------
# Show histograms
# -----------------------------
cv2.imshow("Image with ROIs", img)
cv2.imshow("ROI 1", roi1)
cv2.imshow("ROI 2", roi2)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.hist(roi1.ravel(),256,[0,256])
plt.title("ROI 1 Histogram")

plt.subplot(1,2,2)
plt.hist(roi2.ravel(),256,[0,256])
plt.title("ROI 2 Histogram")

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()