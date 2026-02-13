import cv2

img = cv2.imread("test.jpg")

img2 = cv2.putText(
    img.copy(),
    "Annotated Image",
    (50, 50),                      # (column, row)
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (128, 128, 255),               # BGR colour
    2,
    cv2.LINE_AA
)

cv2.imshow("Annotated", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
