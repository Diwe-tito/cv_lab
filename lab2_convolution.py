import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
# Add salt and pepper noise
noisy = img.copy()

prob = 0.02  # 2% noise

random_matrix = np.random.rand(*img.shape)

noisy[random_matrix < prob/2] = 0        # pepper
noisy[random_matrix > 1 - prob/2] = 255  # salt

if img is None:
    print("Image not found")
    exit()



filtered = np.zeros_like(noisy)

for i in range(1, noisy.shape[0] - 1):
    for j in range(1, noisy.shape[1] - 1):

        region = noisy[i-1:i+2, j-1:j+2]
        median_value = np.median(region)
        filtered[i, j] = median_value

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(noisy, cmap='gray')
plt.title("Salt & Pepper Noise")

plt.subplot(1,3,3)
plt.imshow(filtered, cmap='gray')
plt.title("Median Filtered")

plt.show()