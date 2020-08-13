# All imports
import cv2

# Initialize path
img_path = './Image_set/image2.webp'
car_classifier_path = './cars.xml'
pedestrian_classifier_path = './haarcascade_fullbody.xml'

# load video
video = cv2.VideoCapture('./Test_videos/test_video1.mp4')

# load classifier
car_classifier = cv2.CascadeClassifier(car_classifier_path)
pedestrian_classifier = cv2.CascadeClassifier(pedestrian_classifier_path)

# Process Video
while True:
  # read frames
  (success, frame) = video.read()

  # if success?
  if success:
    # Frame conversion-- RGB to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  else:
    break
  
  # object detection
  cars = car_classifier.detectMultiScale(grayscale_frame)
  pedestrian = pedestrian_classifier.detectMultiScale(grayscale_frame)
  
  # Draw boxes on the frames for cars
  for (x, y, w, h) in cars:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
    cv2.rectangle(frame, (x+1,y+1), (x+w, y+h), (255,0,0), 2)

  # Draw boxes on the frames for pedestrian
  for (x, y, w, h) in pedestrian:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.rectangle(frame, (x+1,y+1), (x+w, y+h), (0,255,255), 2)

  # Display frames
  cv2.imshow('Output', frame)
  
  # Wait for 1ms
  key = cv2.waitKey(1)

  # Loop break
  if key == 13 or key == 32:
    break

#release the video
video.release()

print('Thats it...keep learning...')
print(':) \t :) \t :) ')