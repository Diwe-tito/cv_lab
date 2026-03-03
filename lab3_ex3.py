import cv2
import numpy as np
import matplotlib.pyplot as plt 

# Load colour image
img = cv2.imread("test.jpg", cv2.IMREAD_COLOR)

if img is None:
    print("Image not found")
    exit()

# Convert BGR (OpenCV default) → RGB (for correct plotting)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create empty output image
new_img = np.zeros(img.shape, np.uint8)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):

        # If red channel is very high
        if img[i,j,0] > 150 and img[i,j,1] < 100 and img[i,j,2] < 100:
           new_img[i,j] = [255, 0, 0]

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(new_img)
plt.title("Red Pixels Only")

plt.show()