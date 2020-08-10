#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

from openvino.inference_engine import IECore
import numpy as np
import argparse

import os
import csv
from time import perf_counter

import nibabel as nib

from tqdm import tqdm

from prettytable import PrettyTable

parser = argparse.ArgumentParser(
    description="Quantizes an OpenVINO model to INT8.",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--manifest", default="data/manifest_subset.csv",
                help="Manifest file (CSV with filenames of images and labels)")
parser.add_argument("--data_directory", default="data",
                help="Root directory for the data")

parser.add_argument("-m", "--openvino_model", default="openvino_model/FP32/3d_unet.xml",
                help="OpenVINO XML model filename")

args = parser.parse_args()

class bcolors:
    """
    Just gives us some colors for the text
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def dice_score(pred, truth):
    """
    Sorensen Dice score
    Measure of the overlap between the prediction and ground truth masks
    """
    numerator = np.sum(np.round(pred) * truth) * 2.0
    denominator = np.sum(np.round(pred)) + np.sum(truth)

    return numerator / denominator

def crop_img(img, msk, crop_dim, n_channels, n_out_channels):
    """
    Crop the image and mask
    """

    number_of_dimensions = len(crop_dim)

    slices = []

    for idx in range(number_of_dimensions):  # Go through each dimension

        cropLen = crop_dim[idx]
        imgLen = img.shape[idx]

        start = (imgLen-cropLen)//2

        slices.append(slice(start, start+cropLen))

    # No slicing along channels
    slices_img = slices.copy()
    slices_msk = slices.copy()

    slices_img.append(slice(0, n_channels))
    slices_msk.append(slice(0, n_out_channels))

    return img[tuple(slices_img)], msk[tuple(slices_msk)]

def z_normalize_img(img):
    """
    Normalize the image so that the mean value for each image
    is 0 and the standard deviation is 1.
    """
    for channel in range(img.shape[-1]):

        img_temp = img[..., channel]
        img_temp = (img_temp - np.mean(img_temp)) / np.std(img_temp)

        img[..., channel] = img_temp

    return img

def read_csv_file(filename):
    """
    Read the CSV file with the image and mask filenames
    """
    imgFiles = []
    mskFiles = []
    with open(filename, "r") as f:
        data = csv.reader(f)
        idx = 0
        for row in data:
            if (len(row) > 0) and (idx > 0): # skip header row and empty rows
                imgFiles.append(os.path.join(args.data_directory, row[0]))
                mskFiles.append(os.path.join(args.data_directory, row[1]))
            idx += 1

    return imgFiles, mskFiles, len(imgFiles)

def load_data(imgFile, mskFile, crop_dim, n_channels, n_out_channels, openVINO_order=True):
    """
    Modify this to load your data and labels
    """

    imgs = np.empty((len(imgFile),*crop_dim,n_channels))
    msks = np.empty((len(mskFile),*crop_dim,n_out_channels))
    fileIDs = []

    for idx in range(len(imgFile)):

        img_temp = np.array(nib.load(imgFile[idx]).dataobj)
        msk = np.array(nib.load(mskFile[idx]).dataobj)

        if n_channels == 1:
            img = img_temp[:, :, :, [0]]  # FLAIR channel
        else:
            img = img_temp

        # Add channels to mask
        msk[msk > 0] = 1.0
        msk = np.expand_dims(msk, -1)


        # Crop the image to the input size
        img, msk = crop_img(img, msk, crop_dim, n_channels, n_out_channels)

        # z-normalize the pixel values
        img = z_normalize_img(img)

        fileIDs.append(os.path.basename(imgFile[idx]))

        imgs[idx] = img
        msks[idx] = msk

    if openVINO_order:
        imgs = imgs.transpose((0, 4, 1, 2, 3))
        msks = msks.transpose((0, 4, 1, 2, 3))

    return imgs, msks, fileIDs

ie = IECore()

def load_model(model_xml, fp16=False):
    """
    Load the OpenVINO model.
    """
    print("Loading U-Net model to the plugin")

    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    return model_xml, model_bin

# Read IR
# If using MYRIAD then we need to load FP16 model version
model_xml, model_bin = load_model(args.openvino_model)
print("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
net = ie.read_network(model=model_xml, weights=model_bin)

"""
Ask OpenVINO for input and output tensor names and sizes
"""
input_blob = next(iter(net.inputs))  # Name of the input layer
out_blob = next(iter(net.outputs))   # Name of the output layer

# Load data
batch_size, n_channels, height, width, depth = net.inputs[input_blob].shape
batch_size, n_out_channels, height_out, width_out, depth_out = net.outputs[out_blob].shape
crop_dim = [height, width, depth]

"""
Load the data for OpenVINO
"""
"""
Read the CSV file with the filenames of the images and masks
"""
print("Load data in Numpy array from {}.".format(args.manifest))
print(" (This could take a while if you have lots of data files in the manifest.)")
imgFiles, mskFiles, num_imgs = read_csv_file(args.manifest)
input_data, label_data_ov, img_indicies = load_data(imgFiles, mskFiles,
            crop_dim, n_channels, n_out_channels, openVINO_order=True)

# Reshape the OpenVINO network to accept the different image input shape
# NOTE: This only works for some models (e.g. fully convolutional)
batch_size = 1
n_channels = input_data.shape[1]
height = input_data.shape[2]
width = input_data.shape[3]
depth = input_data.shape[4]

net.reshape({input_blob:(batch_size,n_channels,height,width,depth)})
batch_size, n_channels, height, width, depth = net.inputs[input_blob].shape
batch_size, n_out_channels, height_out, width_out, depth_out = net.outputs[out_blob].shape

print("The network inputs are:")
for idx, input_layer in enumerate(net.inputs.keys()):
    print("{}: {}, shape = {} [N,C,H,W,D]".format(idx,input_layer,net.inputs[input_layer].shape))

print("The network outputs are:")
for idx, output_layer in enumerate(net.outputs.keys()):
    print("{}: {}, shape = {} [N,C,H,W,D]".format(idx,output_layer,net.outputs[output_layer].shape))

# Loading model to the plugin
print("Loading model to the plugin")
exec_net = ie.load_network(network=net, device_name="CPU")
del net

"""
OpenVINO inference code
input_blob is the name (string) of the input tensor in the graph
out_blob is the name (string) of the output tensor in the graph
Essentially, this looks exactly like a feed_dict for TensorFlow inference
"""
# Go through the sample validation dataset to plot predictions
predictions_ov = np.zeros((num_imgs, n_out_channels,
                        depth_out, height_out, width_out))

print("Starting OpenVINO inference")
results = {}
ov_times = []
dice_scores = []

for idx in tqdm(range(0, num_imgs)):

    start_time = perf_counter()

    res = exec_net.infer(inputs={input_blob: input_data[[idx],:n_channels]})

    elapsed_time = 1000.0 * (perf_counter() - start_time)  # msec
    predictions_ov[idx, ] = res[out_blob]
    dice = dice_score(res[out_blob],label_data_ov[idx])
    dice_scores.append(dice)

    results[imgFiles[idx]] = {"time (msec)": elapsed_time, "dice_score": dice}
    ov_times.append(elapsed_time)
    #print("File {}, Dice score = {:.4f}".format(imgFiles[idx], dice_score(res[out_blob],label_data_ov[idx])))

print("Finished OpenVINO inference")
results_table = PrettyTable(["File", "Inference time (msec)", "Dice score"])
for key, item in results.items():
    results_table.add_row([key,
                            "{:.4f}".format(item["time (msec)"]),
                            "{:.4f}".format(item["dice_score"])])

print(bcolors.OKBLUE)
print(results_table)
print(bcolors.ENDC)

print(bcolors.BOLD + "OpenVINO mean times for model {}".format(args.openvino_model) + bcolors.ENDC)
print(bcolors.BOLD + "Mean inference time = {:.4f} msec (s.d. = {:.4f} msec)".format(np.mean(ov_times), np.std(ov_times)) + bcolors.ENDC)
print(bcolors.BOLD + "Mean FPS = {:.4f} scans/sec".format(1000.0/np.mean(ov_times)) + bcolors.ENDC)
print(bcolors.BOLD + "Mean Dice score for subset = {:.4f}".format(np.mean(dice_scores)) + bcolors.ENDC)
