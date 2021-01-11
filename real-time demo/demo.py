import threading

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# global variables
gloss_show = 'Word: none'
labels = {
    0: 'book',
    1: 'chair',
    2: 'clothes',
    3: 'computer',
    4: 'drink',
    5: 'drum',
    6: 'family',
    7: 'football',
    8: 'go',
    9: 'hat',
    10: 'hello',
    11: 'kiss',
    12: 'like',
    13: 'play',
    14: 'school',
    15: 'street',
    16: 'table',
    17: 'university',
    18: 'violin',
    19: 'wall'
}


def main():
    # settings
    dim = (224, 224)
    frames = 10
    channels = 3
    model_path = './model/WLASL20c_model.h5'
    threshold = .50

    print("ASL Real-time Recognition\n")
    print("[INFO] initializing ...")
    # define empty buffer
    frame_buffer = np.empty((0, *dim, channels))

    print("[INFO] loading ASL detection model ...")
    # load model
    model = load_model(model_path)

    print("[INFO] starting video stream ...")
    # start the video stream
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 25)  # set the FPS to 25

    x = threading.Thread()
    # loop over the frames
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # process the frame
            frame_res = cv2.resize(frame, dim)
            frame_res = frame_res / 255.0
            # append the frame to buffer
            frame_resh = np.reshape(frame_res, (1, *frame_res.shape))
            frame_buffer = np.append(frame_buffer, frame_resh, axis=0)
            # start sign recognition only if the buffer is full
            if frame_buffer.shape[0] == frames:
                # make the prediction
                if not x.is_alive():
                    x = threading.Thread(target=make_prediction, args=(
                        frame_buffer, model, threshold))
                    x.start()
                else:
                    pass
                # left-shift of the buffer
                frame_buffer = frame_buffer[1:frames]
                # show label
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, gloss_show, (20, 450), font, 1, (0, 255, 0),
                            2, cv2.LINE_AA)
                cv2.imshow('frame', frame)

            # press Q to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()


def make_prediction(frame_buffer, model, threshold):
    global gloss_show

    frame_buffer_resh = frame_buffer.reshape(1, *frame_buffer.shape)
    # model prediction
    predictions = model.predict(frame_buffer_resh)[0]
    # get the best prediction
    best_pred_idx = np.argmax(predictions)
    acc_best_pred = predictions[best_pred_idx]
    # check mislabeling
    if acc_best_pred > threshold:
        gloss = labels[best_pred_idx]
        gloss_show = "Word: {: <3}  {:.2f}% ".format(
            gloss,
            acc_best_pred * 100)
        print(gloss_show)
    else:
        gloss_show = 'Word: none'


if __name__ == '__main__':
    main()
