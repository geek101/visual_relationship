"""
Helper API
"""

import argparse
import csv
import pickle
import os
import sys
import random
import math
import re
import time
import wget
import numpy as np
from PIL import Image
import cv2
import glob
from shutil import copyfile
import collections
from joblib import Parallel, delayed
from tqdm import tqdm_notebook as tqdm
import parallel 
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_fid(image_path):
        return os.path.basename(image_path).split('.')[0]

def process_labels_from_csv_input(prefix='data/raw', 
                                  labels_csv_fname_list=['class-descriptions-boxable.csv', 
                                                         'class-descriptions.csv']):
    label_dict = {}
    # Process the labels info files and create a key value pair
    for flabel_csv in labels_csv_fname_list:
        with open(os.path.join(prefix, flabel_csv)) as f:
            rows = csv.reader(f)
            for row in rows:
                if row[0] in label_dict:
                    #print('Label already exists: {}:{}, new label: {}:{}'.
                    #       format(row[0], label_dict[row[0]], row[0], row[1]))
                    assert row[1] == label_dict[row[0]]
                label_dict[row[0]]=row[1]
                
    return label_dict
                
def process_raw_csv_input(prefix='data/raw', train_csv_fname = 'challenge-2018-train-vrd.csv', 
                          labels_csv_fname_list = ['class-descriptions-boxable.csv', 'class-descriptions.csv']):
    """
    Process the labels from given label names and create three categories
    a. Entity
    b. Attribute
    c. Relationship
    """
    
    label_dict = process_labels_from_csv_input(prefix, labels_csv_fname_list)
    
    # Process the training data and create x, y i.e x->(imageid, (bounding box data)) y->(label1,label2,relationship)
    label1_dict = collections.defaultdict(int)
    label2_dict = collections.defaultdict(int)
    relationship_dict = collections.defaultdict(int)
    xy_list = []
    missing_label_dict = {}
    ignore_header = True
    miss_count = 0
    with open(os.path.join(prefix, train_csv_fname)) as f:
        rows = csv.reader(f)
        for row in rows:
            miss = False
            if ignore_header:
                ignore_header = False
                continue
            x = (row[0], (row[3:11]))
            y = (row[1], row[2], row[11])
            if y[0] not in label_dict:
                if y[0] not in missing_label_dict:
                    print('Label1 missing: {}'.format(y[0]))
                miss = True
                missing_label_dict[y[0]] = y[0]
            else:
                label1_dict[y[0]] += 1
            if y[1] not in label_dict:
                if y[1] not in missing_label_dict:
                    print('Label2 missing: {}, label1 : {}, relation: {}'.format(y[1], label_dict[y[0]], row[11]))
                miss_count += 1
                miss = True
                missing_label_dict[y[1]] = y[1]
            else:
                label2_dict[y[1]] += 1
            relationship_dict[y[2]] += 1
            if miss is False:
                xy_list.append((x, y))
    print ("Missing label count: {}".format(miss_count))             
    return xy_list, (label1_dict, label2_dict, relationship_dict), label_dict

def get_data_dir_from_raw_single_dir(X_dict, prefix='data', dir_list=None, out_dir='processed'):
    X_fset = set()
    copy_prefix_dir = os.path.join(prefix, out_dir)
    for d in dir_list:
        copy_dir = os.path.join(os.getcwd(), os.path.join(copy_prefix_dir, d))
        os.makedirs(copy_dir, exist_ok=True)
        flist = glob.glob(os.path.join(os.path.join(prefix, d), '*.jpg'))
        for f in flist:
            fid = os.path.basename(f).split('.')[0]
            if fid in X_dict:
                dst_f = os.path.join(copy_dir, os.path.basename(f))
                X_fset.add(dst_f)
                X_dict[fid] = (X_dict[fid], dst_f)
                copyfile(os.path.join(os.getcwd(), f), dst_f)
                
    return X_fset

def get_data_from_dir_recursive(xy_list, prefix='data/processed', dir_input='raw'):
    """
    Load the file path for each image id. The dictionary can only have image file path since
    a single image can have multiple labels i.e multiple y values.

    Example of entry in xy_list:
    Train data xy_list[0]: (
                            ('fe58ec1b06db2bb7', ['0.005', '0.033125', '0.58', '0.62777776', 
                                                   '0.005', '0.033125', '0.58', '0.62777776']) , 
                            ('/m/04bcr3', '/m/083vt', 'is'))
    """
    cwd = os.getcwd()
    xy_list_valid = []      # xy_list that has valid image files available
    X_id_to_file_dict = {}  # id of the image to file dictionary
    def process_files(dir_path):
        flist = glob.glob(os.path.join(dir_path, '*.jpg'))
        print('Processing dir: {}, image count: {}'.format(dir_path, len(flist)))
            
        for f in flist:
            fid = os.path.basename(f).split('.')[0]
            if fid in X_id_to_file_dict:
                print ('Error id exists twice: {}-{}-{}'.format(fid, f, X_id_to_file_dict[fid]))
                continue
            else:
                X_id_to_file_dict[fid] = os.path.join(cwd, f)
                
    def helper(dir_input_full):
        l = next(os.walk(dir_input_full))[1]
        if len(l) == 0:   
            return
        
        for d in l:
            dir_path = os.path.join(dir_input_full, d)
            process_files(dir_path)
            helper(dir_path)
    
    process_files(os.path.join(prefix, dir_input))
    helper(os.path.join(prefix, dir_input))
    
    for xy in xy_list:
        if xy[0][0] in X_id_to_file_dict:
            xy_list_valid.append(xy)

    return xy_list_valid, X_id_to_file_dict

def bounding_box_to_plt(image, b):
    """
    Convert one bounding box data into what mathplotlib understands
    [XMin1,    XMax1,     YMin1,   YMax1,        XMin2,    XMax2,    YMin2,   YMax2]
    ['0.005', '0.033125', '0.58', '0.62777776', '0.005', '0.033125', '0.58', '0.62777776']
    for: https://matplotlib.org/api/_as_gen/matplotlib.patches.Rectangle.html#matplotlib.patches.Rectangle
    """
    xsize = image.shape[1]
    ysize = image.shape[0]
    xy = (int(float(b[0]) * xsize), int(float(b[2]) * ysize))   # (XMin1 * xsize, YMin1 * ysize)
    width = int(float(b[1]) * xsize) - xy[0]        # XMax1 * xsize - XMin1 * xsize
    height = int(float(b[3]) * ysize) - xy[1]       # YMax1 * ysize - Ymin * ysize 
    return (xy, width, height)

def two_bounding_boxes_to_plt(image, b):
    """
    Convert two bounding box data into what mathplotlib understands
    """
    return [bounding_box_to_plt(image, b[0:4]), bounding_box_to_plt(image, b[4:len(b)])]
    
def show_images(images,titles=None, bounding_boxes_list=[]):
    """Display a list of images"""
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    
    for i in range(0, len(images)):
        image = images[i]
        title = "None"
        if titles is not None and len(titles) > i:
            title = titles[i]
        
        bounding_boxes = None
        if bounding_boxes_list is not None and len(bounding_boxes_list) > i:
            bounding_boxes = bounding_boxes_list[i]

        a = fig.add_subplot(1,n_ims,n) # Make subplot
        if len(image.shape) == 2 or image.shape[2] == 1: # Is image grayscale?
            plt.imshow(np.resize(image, (image.shape[0], image.shape[1])), interpolation="bicubic", cmap="gray") # Only place in this blog you can't replace 'gray' with 'grey'
        else:
            plt.imshow(image, interpolation="bicubic")
            if bounding_boxes is not None:
                box1, box2 = two_bounding_boxes_to_plt(image, bounding_boxes)
                rect1 = patches.Rectangle((box1[0]),box1[1],box1[2],linewidth=2,edgecolor='y',facecolor='none')
                rect2 = patches.Rectangle((box2[0]),box2[1],box2[2],linewidth=2,edgecolor='g',facecolor='none')
                a.add_patch(rect1)
                a.add_patch(rect2)
        if titles is not None:
            a.set_title(title + ' {}x{}'.format(image.shape[0], image.shape[1]))
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.axis('off')
    plt.show()
    
def show_given_images(xy_given_list, id_to_file_dict, label_dict):
    img_list = []
    label_list = []
    bounding_boxes_list = []
    for xy in xy_given_list:
        fid = xy[0][0]
        bounding_boxes_list.append(xy[0][1])
        y = xy[1]
        label1 = y[0]
        label2 = y[1]
        if label1 in label_dict:
            label1 = label_dict[label1]
        if label2 in label_dict:
            label2 = label_dict[label2]
        
        label_list.append('{} {} {}'.format(label1, y[2], label2))
        if fid not in id_to_file_dict:
            print ('Error could not find id: {} in id_to_file_dict'.format(fid))
            raise 
        img_list.append(cv2.cvtColor(cv2.imread(id_to_file_dict[fid], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    print ('Label_list" {}'.format(label_list))
    show_images(img_list, titles=label_list, bounding_boxes_list=bounding_boxes_list)
    
def show_random_images(xy_given_list, id_to_file_dict, label_dict, count=4):
    xy_rnd_idx_list = np.random.choice(len(xy_given_list), count, replace=False)
    xy_rnd_list = [ xy_given_list[x] for x in xy_rnd_idx_list]
    show_given_images(xy_rnd_list, id_to_file_dict, label_dict)
    return xy_rnd_list

def _resize_job_helper(kv_list, output_dir, xsize=514, ysize=343):
    out_list = []
    for k, v in kv_list:
        if v is None:
            out_list.append((k, v))
            continue
        try:
            if os.path.isfile(v) is False:
                print('Invalid file failed for {}'.format(v))
                out_list.append((k, v))
                continue
        except:
            print('Invalid file failed for {}'.format(v))
            raise
        out_file = os.path.join(output_dir, os.path.basename(v))
        
        # If the file exists then 
        if os.path.isfile(out_file):
            out_list.append((k, out_file))
            continue
 
        resize_img = cv2.resize(cv2.imread(v, cv2.IMREAD_COLOR),(xsize, ysize))
        out_file = os.path.join(output_dir, os.path.basename(v))
        cv2.imwrite(out_file, resize_img)
        
        out_list.append((k, out_file))
    return out_list

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def resize_all(id_to_file_dict, prefix='data/processed', output_dir='resized_images', xsize=514, ysize=343, 
               count=None):
    output_dir = os.path.join(os.getcwd(), os.path.join(prefix, output_dir))
    os.makedirs(output_dir, exist_ok=True)

    aprun = parallel.ParallelExecutor(n_jobs=7)
    
    chunked_list = list(chunks(list(id_to_file_dict.items()), int(len(id_to_file_dict.items())/8)))
    
    print ('chunk list size: {}'.format(len(chunked_list)))

    out_list = aprun(bar='tqdm')(delayed(_resize_job_helper)(kv_list, output_dir, xsize, ysize) 
                                          for kv_list in chunked_list)
    ret_dict = dict()
    for l in out_list:
        for k, v in l:
            ret_dict[k] = v
    return ret_dict
