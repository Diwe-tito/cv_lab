import cv2

img = cv2.imread("test.jpg")

if img is None:
    print("Image did not load")
else:
    print("Image loaded successfully")
    print("Shape:", img.shape)

    saved = cv2.imwrite("saved_image.jpg", img)

    if saved:
        print("Image saved successfully")
    else:
        print("Image failed to save")
