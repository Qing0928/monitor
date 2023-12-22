import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import time
#----init setting----
#monitor
monitor = cv2.VideoCapture("rtsp://admin:@admin888@192.168.11.197:554")
video_width = monitor.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = monitor.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(video_width, video_height)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video_writer = cv2.VideoWriter('detect_video.mp4', fourcc, 24.0, (1280, 720))
#import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
'''# STEP 3: Load the input image.
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

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='monitor/pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    result_callback=async_result,
    output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)

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
while(True):
    if(monitor.isOpened == False):
        monitor.open()
    ret, frame = monitor.read()
    if(ret == False):
        print("mointor ERROR")
        break
    else:
        pt1 = (825, 230) #左上
        pt2 = (970, 340) #右下
        #cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
        #patch = frame[230:340, 825:970, :]
        #patch = cv2.resize(patch,(200, 120))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(image)
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        video_writer.write(annotated_image)
        cv2.imshow("mointor", annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

'''
class PoseLandmarker_result():
  def __init__(self):
    self.result = None
    self.landmarker = mp.tasks.vision.PoseLandmarker
    self.initLandmarker()

  def initLandmarker(self):
    def updateResult(result:mp.tasks.vision.PoseLandmarkerResult, output_image:mp.Image, timestamp:int):
      self.result = result
      

    options = mp.tasks.vision.PoseLandmarkerOptions(
       base_options = mp.tasks.BaseOptions(model_asset_path = 'monitor/pose_landmarker_lite.task'),
       running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM,
       result_callback = updateResult)
    
    self.landmarker = self.landmarker.create_from_options(options)

  def detect_async(self, frame):
     image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
     self.landmarker.detect_async(image=image, timestamp_ms=int(time.time()*1000))
  
  def close(self):
     self.landmarker.close()

def draw_landmarks_on_image(rgb_image, detect_result):
  #try:
    #print(detect_result)
    if detect_result.pose_landmarks == []:
        return rgb_image
    else:
      pose_landmarks_list = detect_result.pose_landmarks
      #detect_async已經轉換過圖片格式
      #rgb_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
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
  #except Exception as e:
       #print(e)
       #return rgb_image
  
pose_land_marker = PoseLandmarker_result()

while(True):
    if(monitor.isOpened == False):
        monitor.open()
    ret, frame = monitor.read()
    if(ret == False):
        print("mointor ERROR")
        break
    else:
        pose_land_marker.detect_async(frame)
        if hasattr(pose_land_marker.result, 'pose_landmarks') == True:
          #print(pose_land_marker.result.pose_landmarks)
          
          annotated_image = draw_landmarks_on_image(frame, pose_land_marker.result)
          video_writer.write(annotated_image)

          cv2.imshow("mointor", annotated_image)
        else:
           print("fail")
        
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

pose_land_marker.close()
monitor.release()
video_writer.release()
cv2.destroyAllWindows()