import itertools
import os

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class VideoFrameGenerator(keras.utils.Sequence):
    """
    The VideoFrameGenerator extracts n frames from a video.

    You can choose between two sampling mode:
    - Mode 1: extracts n frames from the entire video.
    - Mode 2: extracts n frames from the entire video, but first performs a
    pre-sampling to eliminate the initial and final sequences in witch no
    movements appear.
    """

    def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32),
                 n_channels=3, n_sequence=10, shuffle=True, type_gen='train'):
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_sequence = n_sequence  # number of frames to extract
        self.shuffle = shuffle
        self.type_gen = type_gen
        self.sampl_mode = '2'
        self.aug_gen = ImageDataGenerator()
        print(f'Videos: {len(self.list_IDs)}, batches per epoch: '
              f'{int(np.floor(len(self.list_IDs) / self.batch_size))}')
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def frame_sampling(self, len_frames):
        """
        Sample the video with the chosen policy.

        :param len_frames: video length (in frames)
        :return: the indexes of the sampled frames.
        """
        # create a list of frames
        frames = list(range(len_frames))

        # sampling choice
        if self.sampl_mode == '1':
            # create chunks
            chunks = list(self.get_chunks(frames, self.n_sequence))
            sampling = self.sampling_mode_1(chunks)
        elif self.sampl_mode == '2':
            sampling = self.sampling_mode_2(frames, self.n_sequence)
        else:
            raise ValueError

        return sampling

    def sampling_mode_1(self, chunks):
        """
        Select 10 frames from the entire sequence.

        :param chunks: blocks from which to select frames.
        :return: the indexes of the 10 frames sampled.
        """
        sampling = []
        for i, chunk in enumerate(chunks):
            if i == 0 or i == 1:
                sampling.append(chunk[-1])  # get the last frame
            elif i == (len(chunks) - 1) or i == (len(chunks) - 2):
                sampling.append(chunk[0])  # get the first frame
            else:
                sampling.append(chunk[len(chunk) // 2])  # get the central frame

        return sampling

    def sampling_mode_2(self, frames, n_sequence):
        """
        Do a pre-sampling by creating 12 chunks. Then apply the first sampling
        mode (sampling_mode_1 function).

        :param frames: list of frames.
        :param n_sequence: number of frames to sample.
        :return: the indexes of the 10 frames sampled.
        """
        # create 12 chunks
        chunks = list(self.get_chunks(frames, 12))

        # remove the first and the last chunk
        sub_chunks = chunks[1:-1]

        # get a the new list of frames
        sub_frame_list = list(itertools.chain.from_iterable(sub_chunks))

        # create n_sequence(10) chunks
        new_chunks = list(self.get_chunks(sub_frame_list, n_sequence))

        sampling = self.sampling_mode_1(new_chunks)

        return sampling

    def get_chunks(self, l, n):
        # divide indexes list in n chunks
        k, m = divmod(len(l), n)
        return (l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in
                range(n))

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        # Initialization
        X = np.empty((self.batch_size, self.n_sequence, *self.dim,
                      self.n_channels))
        Y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):  # ID: path to file
            path_file = ID
            cap = cv2.VideoCapture(path_file)
            # get number of frames
            length_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            index_sampling = self.frame_sampling(length_file)  # sampling indxs

            for j, n_pic in enumerate(index_sampling):
                cap.set(cv2.CAP_PROP_POS_FRAMES, n_pic)  # jump to that index
                ret, frame = cap.read()
                new_image = cv2.resize(frame, self.dim)
                X[i, j, :, :, :] = new_image

            if self.type_gen == 'train':
                X[i,] = self.sampling_augmentation(X[i,]) / 255.0
            else:
                X[i,] = X[i,] / 255.0

            Y[i] = self.labels[ID]
            cap.release()

            # for debugging
            # self.save_frame_sampling(X, i, ID, len(index_sampling))

        return X, Y

    def sampling_augmentation(self, sequence):
        """
        - `'theta'`: Float. Rotation angle in degrees.
        - `'tx'`: Float. Shift in the x direction. - vertical shift (height)
        - `'ty'`: Float. Shift in the y direction. - horizontal shift (width)
        - `'shear'`: Float. Shear angle in degrees.
        - `'zx'`: Float. Zoom in the x direction. - vertical zoom
        - `'zy'`: Float. Zoom in the y direction. - horizontal zoom
        - `'flip_horizontal'`: Boolean. Horizontal flip.
        - `'flip_vertical'`: Boolean. Vertical flip.
        - `'channel_shift_intensity'`: Float. Channel shift intensity.
        - `'brightness'`: Float. Brightness shift intensity.
        """
        transformations = ['theta', 'tx', 'ty', 'zx', 'zy', 'flip_horizontal',
                           'brightness']

        # random choice of number of transformations
        random_transforms = np.random.randint(2, 4)  # min 2 - max 3
        # random choice of transformations
        transforms_idxs = np.random.choice(len(transformations),
                                           random_transforms, replace=False)

        transfor_parameters = {}
        for idx in transforms_idxs:
            if transformations[idx] == 'theta':
                transfor_parameters['theta'] = np.random.randint(-5, 5)

            elif transformations[idx] == 'tx':
                transfor_parameters['tx'] = np.random.randint(-10, 10)

            elif transformations[idx] == 'ty':
                transfor_parameters['ty'] = np.random.randint(-15, 15)

            elif transformations[idx] == 'zx':
                transfor_parameters['zx'] = np.random.uniform(0.6, 1.05)

            elif transformations[idx] == 'zy':
                transfor_parameters['zy'] = np.random.uniform(0.6, 1.05)

            elif transformations[idx] == 'flip_horizontal':
                transfor_parameters['flip_horizontal'] = True

            elif transformations[idx] == 'brightness':
                transfor_parameters['brightness'] = np.random.uniform(0.4, 0.6)

        len_seq = sequence.shape[0]
        for i in range(len_seq):
            sequence[i] = self.aug_gen.apply_transform(sequence[i],
                                                       transfor_parameters)

        return sequence

    def save_frame_sampling(self, samp_imgs, i, img_path, n_frames):
        # concatenate all frames
        train_frame = ()
        for n_f in range(n_frames):
            train_frame = (*train_frame, samp_imgs[i, n_f,] * 255.0)

        # get the train of frame in one image
        full_img = np.concatenate(train_frame, axis=1)

        # info
        img_name = (os.path.split(img_path)[1])[:-4]
        img_label = os.path.split(os.path.split(img_path)[0])[1]

        # save the image
        if not os.path.isdir('./sampling_test/'):
            os.mkdir('./sampling_test/')

        name_file = self.type_gen + '_' + img_label + '_' + img_name
        cv2.imwrite('./sampling_test/' + name_file + '.jpg', full_img)
