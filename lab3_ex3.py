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

mask = (img[:,:,0] > 150) & \
       (img[:,:,1] < 100) & \
       (img[:,:,2] < 100)

# Create empty output image
new_img = np.zeros_like(img)
new_img[mask] = [255, 0, 0]


plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(new_img)
plt.title("Red Pixels Only")

plt.show()