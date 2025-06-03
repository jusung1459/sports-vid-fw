import cv2
cap = cv2.VideoCapture(1)
while True:
   status, photo = cap.read()
   cv2.imshow("Streaming Video", photo)
   if cv2.waitKey(10) == 13:
      break
cv2.destroyAllWindows()