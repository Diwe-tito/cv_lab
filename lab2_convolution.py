import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)


if img is None:
    print("Image not found")
    exit()

kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

output = np.zeros_like(img)

for i in range(1, img.shape[0] - 1):
    for j in range(1, img.shape[1] - 1):

        region = img[i-1:i+2, j-1:j+2]
        value = np.sum(region * kernel)
        output[i, j] = value

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(output, cmap='gray')
plt.title("Blurred (Manual Convolution)")

plt.show()
