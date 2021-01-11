"""
The script allows to extend the video length to 12 frames.

Usage: python video_extender.py param1 param2
 - param1: path to videos (e.g. ./videos/)
 - param2: path to the folder that will contain the extended videos
           (e.g. ./output/)
"""
import glob
import os
import shutil
import sys

import cv2
from natsort import natsorted

# global variables
root_folder = './extracted_frames/'


def main():
    # command line inputs
    videos_path = sys.argv[1]
    output_folder_path = sys.argv[2]

    print('[log] > Creating folders ...\n')
    # create the required folders
    create_folders(output_folder_path)

    print('[log] > Processing videos ...\n')
    # get all videos path
    videos = [vid for vid in glob.glob(videos_path + '/*.mp4')]
    for video in videos:
        # extract frames from videos
        extract_frames(video)
        # extend the video and save it
        extend_video(video, output_folder_path)

    print('[log] > Done!')


def create_folders(output_folder_path):
    # create root folder to contain extracted frames
    if not os.path.isdir(root_folder):
        os.mkdir(root_folder)
    # create the output folder that will contain the extended videos
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)


def extract_frames(video_path):
    # create a folder to store the extracted frames
    create_frames_folder(video_path)
    # extract frames
    video_name = video_path.split('/')[-1][:-4]
    folder_extracted_frames = root_folder + video_name
    vid_cap = cv2.VideoCapture(video_path)
    success, image = vid_cap.read()
    count = 0
    while success:
        # save frame as JPEG file
        cv2.imwrite(folder_extracted_frames + '/%d.jpg' % count, image)
        success, image = vid_cap.read()
        count += 1


def create_frames_folder(video_path):
    video_name = video_path.split('/')[-1][:-4]
    folder_extracted_frames = root_folder + video_name
    if os.path.isdir(folder_extracted_frames):
        shutil.rmtree(folder_extracted_frames)
        os.mkdir(folder_extracted_frames)
    else:
        os.mkdir(folder_extracted_frames)


def extend_video(video_path, output_folder_path):
    video_name = video_path.split('/')[-1][:-4]
    folder_extracted_frames = root_folder + video_name
    # get the extracted frames in order
    frames = [img for img in glob.glob(folder_extracted_frames + '/*.jpg')]
    frames = natsorted(frames)
    # collect frames together
    size = (0, 0)
    img_array = []
    for filename in frames:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    # add padding frames
    padding = 12 - len(img_array)
    for _ in range(padding):
        img_array.append(img_array[-1])
    # save the video
    extended_video_path = output_folder_path + video_name + '.mp4'
    fourcc = 0x7634706d  # mp4 format
    fps = 25
    out = cv2.VideoWriter(extended_video_path, fourcc, fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    # show info
    video_cap = cv2.VideoCapture(extended_video_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    video_cap.release()
    print(f'Video: {video_name}')
    print(f'FPS: {fps}')
    print(f'Frames: {frame_count}')
    print(f'Duration: {duration}s\n')


if __name__ == '__main__':
    main()
