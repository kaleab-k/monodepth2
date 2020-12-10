# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib
import random
import numpy as np

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))

def split_dataset(dataset, path, side, max_idx, min_idx=0, split_ratio=0.8):

    idxs = np.arange(min_idx, max_idx)
    random.shuffle(idxs)
    train_max_idx = int(max_idx * split_ratio)
    train_idxs = idxs[0:train_max_idx]
    test_idxs = idxs[train_max_idx:max_idx]

    with open('splits/'+ dataset +'/train_files.txt', 'a') as train_file:
        for idx in train_idxs:
            train_file.write('{} {} {}\n'.format(path, str(idx), side))

    with open('splits/'+ dataset +'/val_files.txt', 'a') as test_file:
        for idx in test_idxs:
            test_file.write('{} {} {}\n'.format(path, str(idx), side))

def get_files_list(dirName, excludeList): 
 
    listOfFile = os.listdir(dirName) 
    allFiles = list() 
    for entry in listOfFile: 
        fullPath = os.path.join(dirName, entry) 
        if any(substring in fullPath for substring in excludeList):
            continue
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_files_list(fullPath, excludeList) 
        else:
            # fullPath = fullPath.replace("/RGB", "")
            allFiles.append(fullPath) 
    return allFiles

def kitti_split_dataset(dataset, path, side='l', split_ratio=0.8):
    exclude_list = ["velodyne_points", "timestamps.txt", "calibration", "data_poses", "calib_cam_to_cam", "instance"]
    files_list = get_files_list(path, exclude_list)
    max_idx = len(files_list)
    print("Total files: ", max_idx)
    idxs = np.arange(1, max_idx)
    random.shuffle(idxs)
    train_max_idx = int(max_idx * split_ratio)
    train_idxs = idxs[0:train_max_idx]
    test_idxs = idxs[train_max_idx:max_idx]
    for phase, idxs in zip( ["train", "val"], [train_idxs, test_idxs] ):
        print(phase)
        with open('splits/'+ dataset +'/'+ phase +'_files.txt', 'w+') as split_file:
            for idx in idxs:
                cur_file = files_list[idx]
                base = os.path.basename(cur_file) 
                dir_path = os.path.dirname(cur_file)
                total_files = len( os.listdir(dir_path) )

                dir_path = dir_path.replace(path, ".")
                if "image_01" in dir_path:
                    side = 'r'
                    dir_path = dir_path.replace("image_01/data_rect", "")
                else:
                    side = 'l'
                    dir_path = dir_path.replace("image_00/data_rect", "")
                 
                frame_num = int(os.path.splitext(base)[0])

                if frame_num == 0 or (frame_num) == (total_files)-1:
                    continue
                split_file.write('{} {} {}\n'.format(dir_path, (frame_num), side))
