#Usar
#python MTCNN_01_FaceDetection_imagem.py

from mtcnn import MTCNN
import cv2
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4500)])

detector = MTCNN()

imagem = cv2.imread('Eduardo_Marinho.jpg')

faces = detector.detect_faces(imagem)
for face in faces:
    #print(face)
    x, y, width, height = face['box']
    nose = face['keypoints']['nose']
    mouth_right = face['keypoints']['mouth_right']
    mouth_left = face['keypoints']['mouth_left']
    right_eye = face['keypoints']['right_eye']
    left_eye = face['keypoints']['left_eye']
    cv2.rectangle(imagem, (x, y), (x+width, y+height),(0, 0, 255), 2)
    cv2.circle(imagem, (nose), 2, (255, 0, 0), 2)
    cv2.circle(imagem, (mouth_right), 2, (255, 0, 0), 2)
    cv2.circle(imagem, (mouth_left), 2, (255, 0, 0), 2)
    cv2.circle(imagem, (right_eye), 2, (255, 0, 0), 2)
    cv2.circle(imagem, (left_eye), 2, (255, 0, 0), 2)
    print(face['confidence'])
cv2.imshow("Output", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
