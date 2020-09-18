from __future__ import division

import os
from random import shuffle

import numpy as np
from cv2 import VideoCapture
from face_recognition import face_locations
from keras.utils import to_categorical


def _get_videos_path_in_path(path):
    result = []
    for path, _, files in os.walk(path):
        for file in files:
            if file[-3:] in ['avi', 'mp4']:
                result.append(os.path.join(path, file))
    return result


def _get_label(y, classes=None):
    if type(classes) is dict and type(y) is str:
        y = classes[y]

    if classes:
        return to_categorical(y, len(classes))
    else:
        return to_categorical(y)


def generator(dataset_dir, batch_size, shape=(200, 200, 3), classes=None):  # TODO: Classes
    if classes is None:
        classes = {
            "angry": 0,
            "disgust": 1,
            "fear": 2,
            "happy": 3,
            "sadly": 4,
            "surprised": 5
        }

    files = _get_videos_path_in_path(dataset_dir)
    shuffle(files)

    x, y = [], []
    while True:
        for file in files:
            if '._' not in file:
                cap = VideoCapture(file)

                if os.name == 'nt':
                    label_name = file.split('\\')[-2]
                else:
                    label_name = file.split('/')[-2]

                label = to_categorical(classes[label_name], len(classes))

                while True:
                    res, frame = cap.read()

                    if not res:
                        break

                    for top, right, bottom, left in face_locations(frame):
                        image = frame[top:bottom, left:right]

                        image = np.resize(frame, shape) / 255

                        x.append(image)
                        y.append(label)

                        if len(x) == batch_size:
                            yield np.array(x), np.array(y)
                            x, y = [], []


def get_steps_per_epoch(dataset_dir, batch_size=1):
    data = 0
    for path, _, files in os.walk(dataset_dir):
        for file in files:
            if '._' not in file:
                cap = VideoCapture(os.path.join(path, file))

                res = True
                while res:
                    res, _ = cap.read()
                    data += 1
    return data // batch_size


def get_faces(directory, shape=(200, 200, 3)):
    from cv2 import imwrite
    files = _get_videos_path_in_path(directory)
    for file in files:
        print(file)
        cap = VideoCapture(file)

        frame_id = -1
        while True:
            res, frame = cap.read()
            frame_id += 1

            if not res:
                break

            for top, right, bottom, left in face_locations(frame):
                image = frame[top:bottom, left:right]
                image = np.resize(image, shape)

                save_dir = ''.join(file.split('\\')[:-1]) + "\\images"
                save_filename = "\\{}_{}.jpg".format(file.split('.')[0], frame_id)
                imwrite(save_dir + save_filename, image)
