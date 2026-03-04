import cv2
import numpy as np
import matplotlib.pyplot as plt 

# Load image (BGR)
img = cv2.imread("test.jpg")

if img is None:
    print("Image not found")
    exit()

# Convert to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define red ranges (remember red wraps around)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Create masks
mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)

# Combine masks
mask = mask1 + mask2

# Apply mask to original image
result = cv2.bitwise_and(img, img, mask=mask)

# Convert BGR to RGB for plotting
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# Display
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Original")

plt.subplot(1,2,2)
plt.imshow(result_rgb)
plt.title("Red Detected (HSV)")

plt.show()