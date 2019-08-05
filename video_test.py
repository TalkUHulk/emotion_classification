import face_recognition
import cv2 as cv
import sys
import tensorflow as tf
from model_resnet import resnet
import numpy as np

b_Saved = True
b_Show = False
TYPE = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model_path = 'model/resnet/'

train_mode = tf.placeholder(tf.bool)

input = tf.placeholder(tf.float32, shape=[1, 48, 48, 1])
model = resnet(len(TYPE), train_mode)
logits = tf.nn.softmax(
    model.build(input, 2)
)

saver = tf.train.Saver()

cam = cv.VideoCapture('./video_test/emotion_test.mp4')
if not cam.isOpened():
    sys.exit()

if b_Saved:
    width = cam.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = int(cam.get(cv.CAP_PROP_FPS))
    writer = cv.VideoWriter(
        './video_test/emotion_test_result.avi', cv.VideoWriter_fourcc(*'MJPG'), fps, (int(width), int(height))
    )


with tf.Session() as sess:
    if tf.train.latest_checkpoint(model_path) is not None:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
    else:
        assert 'can not find checkpoint folder path!'

    try:
        while True:
            ret, bgr = cam.read()
            if not ret:
                cam.set(cv.CAP_PROP_POS_FRAMES, 0)
                if b_Saved:
                    break
                if b_Show:
                    continue

            h, w, _ = bgr.shape
            #bgr = cv.resize(bgr, (w // 2, h //2))
            gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

            face_locations = face_recognition.face_locations(bgr)
            emotion_list = []
            face_list = []
            for face_location in face_locations:

                top, right, bottom, left = face_location
                face_roi = gray[top:bottom, left:right]
                face_roi = cv.resize(face_roi, (48, 48))
                face_list.append(face_roi)
                logits_ = sess.run(logits,
                                   feed_dict={
                                       input: np.reshape(face_roi.astype(np.float32) / 255.0, (1,) + face_roi.shape + (1,)),
                                       train_mode: False
                                   })

                emotion = TYPE[np.argmax(logits_[0])]
                emotion_list.append(emotion)
                cv.rectangle(bgr, (left, top), (right, bottom), (0, 255, 0), 2)
                cv.rectangle(bgr, (left, top - 20), (right, top), (0, 255, 0), cv.FILLED)
                cv.putText(bgr, emotion, (left, top), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), thickness=1)

            print('detect face:{}, emotion:{}'.format(len(face_locations), emotion_list))
            if b_Show:
                cv.imshow('Camera', bgr)
                for index, roi in enumerate(face_list):
                    cv.imshow('roi_%d' % index, roi)
                cv.waitKey(1)
            if b_Saved:
                writer.write(bgr)
    except Exception as e:
        print('Error:', e)
        sys.exit()
    finally:
        cam.release()
        if b_Saved:
            writer.release()




