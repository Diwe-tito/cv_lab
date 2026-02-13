import cv2
import numpy as np

img = cv2.imread("test.jpg")

b, g, r = cv2.split(img)

# Create images where only one channel is kept
blue_img = cv2.merge([b, np.zeros_like(b), np.zeros_like(b)])
green_img = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])
red_img = cv2.merge([np.zeros_like(r), np.zeros_like(r), r])
resized = cv2.resize(img, (512, 512))

small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
smooth = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
sharp = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

cv2.imshow("Original", img)
cv2.imshow("Blue Only", blue_img)
cv2.imshow("Green Only", green_img)
cv2.imshow("Red Only", red_img)
cv2.imshow("Resized", resized)  
cv2.imshow("Nearest", small)
cv2.imshow("Linear", smooth)
cv2.imshow("Cubic", sharp)

print("Original shape:", img.shape)
print("Resized shape:", resized.shape)

roi = img[200:600, 300:700]
cv2.imshow("ROI", roi)  

cv2.waitKey(0)
cv2.destroyAllWindows()
