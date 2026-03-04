import cv2
import numpy as np

# Load image
img = cv2.imread("test.jpg")

if img is None:
    print("Image not found")
    exit()

# Convert to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow("HSV Threshold Tool", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HSV Threshold Tool", 800, 600)

# Dummy callback (required)
def nothing(x):
    pass

# Create trackbars
cv2.createTrackbar("H Min", "HSV Threshold Tool", 0, 180, nothing)
cv2.createTrackbar("H Max", "HSV Threshold Tool", 180, 180, nothing)
cv2.createTrackbar("S Min", "HSV Threshold Tool", 0, 255, nothing)
cv2.createTrackbar("S Max", "HSV Threshold Tool", 255, 255, nothing)
cv2.createTrackbar("V Min", "HSV Threshold Tool", 0, 255, nothing)
cv2.createTrackbar("V Max", "HSV Threshold Tool", 255, 255, nothing)

while True:

    # Get slider values
    h_min = cv2.getTrackbarPos("H Min", "HSV Threshold Tool")
    h_max = cv2.getTrackbarPos("H Max", "HSV Threshold Tool")
    s_min = cv2.getTrackbarPos("S Min", "HSV Threshold Tool")
    s_max = cv2.getTrackbarPos("S Max", "HSV Threshold Tool")
    v_min = cv2.getTrackbarPos("V Min", "HSV Threshold Tool")
    v_max = cv2.getTrackbarPos("V Max", "HSV Threshold Tool")

    # Define range
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # Create mask
    mask = cv2.inRange(img_hsv, lower, upper)

    # Apply mask
    result = cv2.bitwise_and(img, img, mask=mask)

    # Show result
    display = result.copy()

    text = f"H:{h_min}-{h_max} S:{s_min}-{s_max} V:{v_min}-{v_max}"
    cv2.putText(display, text, (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0,255,0), 2)

    cv2.imshow("HSV Threshold Tool", display)

    # Exit when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()