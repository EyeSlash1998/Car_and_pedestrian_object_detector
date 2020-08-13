#All imports
import cv2

#Initialize path
img_path = './Image_set/image2.webp'
car_classifier_path = './cars.xml'

#load image
img = cv2.imread(img_path)

#load classifier
car_classifier = cv2.CascadeClassifier(car_classifier_path)

#Convert image --RGB to grayscale
b_and_w_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect car
cars = car_classifier.detectMultiScale(b_and_w_img)

#Draw detected boxes on the image
for(x, y, w, h) in cars:
  cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

#Display detected image
cv2.imshow('Detected Output', img)
cv2.waitKey()


print('The end')