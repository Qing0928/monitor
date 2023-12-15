import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
#----init setting----
#monitor
#monitor = cv2.VideoCapture("rtsp://admin:@admin888@192.168.11.197:554")
#import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
'''while(True):
    if(monitor.isOpened == False):
        monitor.open()
    ret, frame = monitor.read()
    if(ret == False):
        print("mointor ERROR")
        break
    else:
        cv2.imshow("mointor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

monitor.release()
cv2.destroyAllWindows()'''



def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    #extend有點類似add，不過是把一個list加進另一個list
    
    pose_landmarks_proto.landmark.extend([
      landmark_pb2
      .NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='monitor/pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image_mat = cv2.imread('monitor/image.jpg')
#print(image_mat.shape)
#image_mat = cv2.resize(image_mat, (300, 400), interpolation=cv2.INTER_AREA)
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_mat)
#image = mp.Image.create_from_file("monitor/test.jpg")

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image) #檢測結果，一堆座標
#print(detection_result)

#把檢測結果畫上去
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
#cv2.imshow("detect_result", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.imshow("detect_result", annotated_image)
cv2.waitKey(0)