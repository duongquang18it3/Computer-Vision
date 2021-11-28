import cv2


# reading the image
img = cv2.imread('book.jpg')
# convert to greyscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# create SIFT feature extractor
# Tao trinh trich xuat dat trung SIFT
sift = cv2.xfeatures2d.SIFT_create()
# detect features from the image
# Phat hien tinh nang tu anh
keypoints, descriptors = sift.detectAndCompute(img, None)
# draw the detected key points
# Rut ve ra cac diem chinh duoc phat hien
sift_image = cv2.drawKeypoints(gray, keypoints, img)
# show the image
cv2.imshow('image', sift_image)
# save the image
cv2.imwrite("table-sift.jpg", sift_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
