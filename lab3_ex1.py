import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found")
    exit()

# 3x3 box filter
kernel = np.ones((7,7), np.float32) / 49

img_conv = cv2.filter2D(img, -1, kernel)

sobel_x = np.array([[1,0,-1],
                    [2,0,-2],
                    [1,0,-1]], dtype=np.float32)

sobel_y = np.array([[1,2,1],
                    [0,0,0],
                    [-1,-2,-1]], dtype=np.float32)

sobel_x_img = cv2.filter2D(img, cv2.CV_32F, sobel_x)
sobel_y_img = cv2.filter2D(img, cv2.CV_32F, sobel_y)

# Compute magnitude
magnitude = np.sqrt(sobel_x_img**2 + sobel_y_img**2)

# Normalize magnitude for display
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
magnitude = magnitude.astype(np.uint8)

# Compute direction (in radians)
direction = np.arctan2(sobel_y_img, sobel_x_img)

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(2,3,2)
plt.imshow(sobel_x_img, cmap='gray')
plt.title("Sobel X")

plt.subplot(2,3,3)
plt.imshow(sobel_y_img, cmap='gray')
plt.title("Sobel Y")

plt.subplot(2,3,4)
plt.imshow(magnitude, cmap='gray')
plt.title("Gradient Magnitude")

plt.subplot(2,3,5)
plt.imshow(direction, cmap='gray')
plt.title("Gradient Direction")

plt.tight_layout()
plt.show()