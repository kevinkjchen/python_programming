import cv2
import numpy as np

img1 = cv2.imread("image/homework2.png", 1)
img_b = img1[:,:,0]
img_g = img1[:,:,1]
img_r = img1[:,:,2]

# #print(img1.shape, img_r.shape)
# img2 = cv2.subtract(img1, m1)
# #img3 = cv2.absdiff(img2, m2)
# img3 = cv2.multiply(img2, m2)

img2 = cv2.subtract(img_r, img_g)
img3 = cv2.subtract(img2, img_b)
img3 = cv2.bitwise_not(img3)

cv2.imshow("Image 1", img1)
#cv2.imshow("Image 2", img2)
cv2.imshow("Image 3", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()