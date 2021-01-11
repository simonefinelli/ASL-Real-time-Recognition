import glob
import os


def labels_to_number(path):
    """
    Convert string labels in numbers.

    :param path: path to folder.
    :return: a dictionary.
    """
    classes = [i.split(os.path.sep)[3] for i in glob.glob(path + '*')]
    classes.sort()

    labels_dict = {}
    for i, label in enumerate(classes):
        labels_dict[label] = i

    return labels_dict


def videos_to_dict(path, labels):
    """
    Read the videos and return a dict like {'path_to_video', 'label'}.

    :param path: path to videos folder.
    :param labels: labels as dict.
    :return: a dictionary.
    """
    videos_dict = {}
    for root, dirs, files in os.walk(os.path.relpath(path)):
        for file in files:
            video_name = os.path.join(root, file)
            dir_name = os.path.basename(os.path.dirname(video_name))  # label
            videos_dict[video_name] = labels[dir_name]

    return videos_dict
