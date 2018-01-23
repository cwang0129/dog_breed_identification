# -*- coding: utf-8 -*-
import pandas as pd
import os
import numpy as np
import cv2

RESIZE_DIMENSION = 500
df = pd.read_csv("data/labels.csv")
labels_dict = dict(zip(df['id'],df['breed']))

int2label = {}
label2int = {}
for line in open("data/label2id.txt"):
	line = line.strip()
	if len(line) > 0:
		label, idd = line.split("\t")
		label2int[label] = int(idd)
		int2label[int(idd)] = label


def load_data(dir_name, keep_original=True):
	# load_data("train")
	global RESIZE_DIMENSION
	images_list = []
	images_ids = []
	label_ids = []
	for file_name in os.listdir(dir_name):
		if not file_name.endswith(".jpg"):
			continue
		file_path = dir_name + "/" + file_name
		cvimage = cv2.imread(file_path)
		resized_image = cv2.resize(cvimage, (RESIZE_DIMENSION, RESIZE_DIMENSION)) 

		file_id = file_name[:-4]

		if not keep_original:
			resized_image = np.reshape(resized_image, (RESIZE_DIMENSION*RESIZE_DIMENSION*3,))
		images_list.append(resized_image)
		images_ids.append(file_id)
		label_ids.append(label2int[labels_dict[file_id]])

	#return np.asarray(images_list).T, images_ids
	return np.asarray(images_list), label_ids

def id2label(image_id):
	global labels_dict
	return labels_dict[image_id]

def ids2label(images_ids):
	ys = []
	for image_id in images_ids:
		ys.append(id2label(image_id))
	return np.asarray(ys).reshape(1, len(ys))

if True:
	label2id = {}
	for label in labels_dict:
		label = labels_dict[label]
		if label in label2id:
			continue
		label2id[label] = len(label2id)
	fout = open("data/label2id.txt", "w")
	for label in label2id:
		fout.write(label + "\t" + str(label2id[label]) + "\n")
	#print label2id

