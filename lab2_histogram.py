import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

#histogram
hist_data = np.zeros(256)
print("Histogram shape:" ,hist_data.shape)
for i in range(img.shape[0]):      # rows
    for j in range(img.shape[1]):  # columns
        pixel_value = img[i, j]
        hist_data[pixel_value] += 1

print("Shape:", img.shape)
print("Total pixels counted:", np.sum(hist_data))
print("Total pixels in image:", img.shape[0] * img.shape[1])
plt.figure()
plt.plot(hist_data)
plt.title("Manual Histogram")
plt.xlabel("Intensity Value (0-255)")
plt.ylabel("Number of Pixels")
plt.show()

#cdf
cdf = np.zeros(256)
cdf[0] = hist_data[0]
for i in range(1, 256):
    cdf[i] = cdf[i-1] + hist_data[i]

print("Last CDF value:", cdf[255])
print("Total pixels:", img.shape[0] * img.shape[1])
plt.figure()
plt.plot(cdf)
plt.title("Cumulative Distribution Function (CDF)")
plt.xlabel("Intensity Value (0-255)")
plt.ylabel("Cumulative Pixel Count")
plt.show()

#equalisation
L = 256
n = img.shape[0] * img.shape[1]
transfer = ((L - 1) / n) * cdf
transfer = transfer.astype(np.uint8)
img_eq = np.zeros_like(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img_eq[i, j] = transfer[img[i, j]]

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(img_eq, cmap='gray')
plt.title("Equalised")

plt.show()



