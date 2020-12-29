#Usar
#python MTCNN_03_FaceDetection_video_fast.py

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

x_resize = 0.15
y_resize = 0.15

while(True):
    ret, frame = cap.read()
    frame_small = cv2.resize(frame, (int(x_resize*frame.shape[1]), int(y_resize*frame.shape[0])), interpolation = cv2.INTER_AREA)
    start_time = time.time()
    faces = detector.detect_faces(frame_small)
    print('Tempo: {}'.format(time.time() - start_time))
    for face in faces:
        #print(face)
        x, y, width, height = face['box']
        x = int(x/x_resize)
        y = int(y/y_resize)
        width = int((width)/x_resize)
        height = int((height)/y_resize)
        cv2.rectangle(frame, (x, y), (x+width, y+height),(0, 0, 255), 5)
        #print(face['confidence'])
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release() 
cv2.destroyAllWindows()
