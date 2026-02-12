import cv2

img = cv2.imread("test.jpg")

if img is None:
    print("Image did not load")
else:
    img[100:150, 100:150] = [0, 200, 0]

    cv2.imshow("Modified Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
    print("Color shape:", img.shape)
    print("Gray shape:", gray.shape)
 