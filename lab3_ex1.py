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

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(2,2,2)
plt.imshow(img_conv, cmap='gray')
plt.title("3x3 Box Filter")

plt.subplot(2,2,3)
plt.imshow(sobel_x_img, cmap='gray')
plt.title("Sobel X")

plt.subplot(2,2,4)
plt.imshow(sobel_y_img, cmap='gray')
plt.title("Sobel Y")

plt.tight_layout()
plt.show()