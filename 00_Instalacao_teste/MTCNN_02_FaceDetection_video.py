#Usar
#python MTCNN_02_FaceDetection_video.py

from mtcnn import MTCNN
import cv2
import tensorflow as tf
import time

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4500)])

detector = MTCNN()

cap = cv2.VideoCapture('EduardoMarinho.mp4')
#cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    start_time = time.time()
    faces = detector.detect_faces(frame)
    print('Tempo: {}'.format(time.time() - start_time))
    for face in faces:
        #print(face)
        x, y, width, height = face['box']
        nose = face['keypoints']['nose']
        mouth_right = face['keypoints']['mouth_right']
        mouth_left = face['keypoints']['mouth_left']
        right_eye = face['keypoints']['right_eye']
        left_eye = face['keypoints']['left_eye']
        cv2.rectangle(frame, (x, y), (x+width, y+height),(0, 0, 255), 2)
        cv2.circle(frame, (nose), 2, (255, 0, 0), 2)
        cv2.circle(frame, (mouth_right), 2, (255, 0, 0), 2)
        cv2.circle(frame, (mouth_left), 2, (255, 0, 0), 2)
        cv2.circle(frame, (right_eye), 2, (255, 0, 0), 2)
        cv2.circle(frame, (left_eye), 2, (255, 0, 0), 2)
        print(face['confidence'])
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release() 
cv2.destroyAllWindows()
