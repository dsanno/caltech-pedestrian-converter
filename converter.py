#!/usr/bin/python
# -*- coding: utf-8 -*-

# This script converts .seq files into .jpg files, .vbb files into .json files
# from Caltech Pedestrian Dataset
# Based on Python 2.7

# Modified by Daiki Sanno

# Original Author: Peng Zhang
# E-mail: hizhangp@gmail.com
# Caltech Pedestrian Dataset:
# http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/

import argparse
import json
import struct
import os
import time
from scipy.io import loadmat
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data',
        help='Input data directory')
    parser.add_argument('--output-dir', type=str, default='data',
        help='Output directory')
    parser.add_argument('--train-interval', type=int, default=5,
        help='Train data frame interval')
    parser.add_argument('--test-interval', type=int, default=30,
        help='Test data frame interval')
    return parser.parse_args()


def read_seq(path, interval=1):
    def read_header(ifile):
        feed = ifile.read(4)
        norpix = ifile.read(24)
        version = struct.unpack('@i', ifile.read(4))
        length = struct.unpack('@i', ifile.read(4))
        assert length != 1024
        descr = ifile.read(512)
        params = [struct.unpack('@i', ifile.read(4))[0] for i in range(9)]
        fps = struct.unpack('@d', ifile.read(8))
        ifile.read(432)
        image_ext = {100: 'raw', 102: 'jpg', 201: 'jpg', 1: 'png', 2: 'png'}
        return {'w': params[0], 'h': params[1], 'bdepth': params[2],
                'ext': image_ext[params[5]], 'format': params[5],
                'size': params[4], 'true_size': params[8],
                'num_frames': params[6]}

    assert path[-3:] == 'seq', path
    ifile = open(path, 'rb')
    params = read_header(ifile)
    bytes = open(path, 'rb').read()

    imgs = []
    extra = 8
    s = 1024
    for i in range(params['num_frames']):
        tmp = struct.unpack_from('@I', bytes[s:s + 4])[0]
        I = bytes[s + 4:s + tmp]
        s += tmp + extra
        if i == 0:
            val = struct.unpack_from('@B', bytes[s:s + 1])[0]
            if val != 0:
                s -= 4
            else:
                extra += 8
                s += 8
        if (i + 1) % interval == 0:
            imgs.append((i, I))

    return imgs


def read_vbb(path):
    assert path[-3:] == 'vbb'
    vbb = loadmat(path)
    nFrame = int(vbb['A'][0][0][0][0][0])
    objLists = vbb['A'][0][0][1][0]
    maxObj = int(vbb['A'][0][0][2][0][0])
    objInit = vbb['A'][0][0][3][0]
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
    objStr = vbb['A'][0][0][5][0]
    objEnd = vbb['A'][0][0][6][0]
    objHide = vbb['A'][0][0][7][0]
    altered = int(vbb['A'][0][0][8][0][0])
    log = vbb['A'][0][0][9][0]
    logLen = int(vbb['A'][0][0][10][0][0])

    data = {}
    data['nFrame'] = nFrame
    data['maxObj'] = maxObj
    data['log'] = log.tolist()
    data['logLen'] = logLen
    data['altered'] = altered
    data['frames'] = defaultdict(list)

    for frame_id, obj in enumerate(objLists):
        if len(obj) > 0:
            for id, pos, occl, lock, posv in zip(obj['id'][0],
                                                 obj['pos'][0],
                                                 obj['occl'][0],
                                                 obj['lock'][0],
                                                 obj['posv'][0]):
                keys = ['id', 'pos', 'occl', 'lock', 'posv']
                id = int(id[0][0]) - 1  # MATLAB is 1-origin
                p = pos[0].tolist()
                pos = [p[0] - 1, p[1] - 1, p[2], p[3]]  # MATLAB is 1-origin
                occl = int(occl[0][0])
                lock = int(lock[0][0])
                posv = posv[0].tolist()

                datum = dict(zip(keys, [id, pos, occl, lock, posv]))
                datum['lbl'] = str(objLbl[datum['id']])
                # MATLAB is 1-origin
                datum['str'] = int(objStr[datum['id']]) - 1
                # MATLAB is 1-origin
                datum['end'] = int(objEnd[datum['id']]) - 1
                datum['hide'] = int(objHide[datum['id']])
                datum['init'] = int(objInit[datum['id']])

                data['frames'][frame_id].append(datum)

    return data

def make_image_annotation(annotations, set_index, seq_index, frame):
    set_label = '{0:02d}'.format(set_index)
    seq_label = '{0:02d}'.format(seq_index)
    image_annotations = annotations[set_label][seq_label]['frames'][frame]
    regions = []
    for annotation in image_annotations:
        if annotation['lbl'] != 'person':
            continue
        x, y, w, h = annotation['pos']
        vx, vy, vw, vh = annotation['posv']
        if vx == 0 and vy == 0 and vw == 0 and vh == 0:
            vx = x
            vy = y
            vw = w
            vh = h
        regions.append({
            'category': 'person',
            'bbox': [x, y, w, h],
            'visible_bbox': [vx, vy, vw, vh]
        })
    return {'regions': regions}


def convert_seq(set_index, input_dir, output_dir, annotations, interval=1):
    set_name = 'set{:02}'.format(set_index)
    img_set_path = os.path.join(input_dir, set_name)
    assert os.path.exists(
        img_set_path), 'Not exists: '.format(img_set_path)
    print('Extracting images from set{:02} ...'.format(set_index))
    set_save_path = os.path.join(output_dir, set_name)
    if not os.path.exists(set_save_path):
        os.mkdir(set_save_path)
    for j in sorted(os.listdir(img_set_path)):
        imgs_path = os.path.join(img_set_path, j)
        imgs = read_seq(imgs_path, interval)
        seq_name = j[:4]
        seq_index = int(j[1:4])
        seq_save_path = os.path.join(set_save_path, seq_name)
        if not os.path.exists(seq_save_path):
            os.mkdir(seq_save_path)
        for ix, img in imgs:
            img_name = '{:05}.jpg'.format(ix)
            img_path = os.path.join(seq_save_path, img_name)
            with open(img_path, 'wb') as f:
                f.write(img)
            annotation_name = '{:05}.json'.format(ix)
            annotation_path = os.path.join(seq_save_path, annotation_name)
            image_annotations = make_image_annotation(annotations, set_index,
                seq_index, ix)
            with open(annotation_path, 'w') as f:
                json.dump(image_annotations, f)


def main():
    args = parse_args()
    # directory to store data
    dir_path = args.data_dir
    output_path = args.output_dir
    # num ranges from 0~11
    num = [0, 11]
    train_range = [0, 6]
    test_range = [6, 11]

    time_flag = time.time()
    train_img_save_path = os.path.join(output_path, 'train_images')
    test_img_save_path = os.path.join(output_path, 'test_images')
    anno_save_path = os.path.join(output_path, 'annotations.json')
    if not os.path.exists(train_img_save_path):
        os.mkdir(train_img_save_path)
    if not os.path.exists(test_img_save_path):
        os.mkdir(test_img_save_path)
    print('Images will be saved to')
    print(train_img_save_path)
    print(test_img_save_path)
    print('Annotations will be saved to {}'.format(anno_save_path))

    # convert .vbb file into .json
    # example: anno['00']['00']['frames'][0][0]['pos']
    anno = defaultdict(dict)
    for i in range(num[0], num[1]):
        anno['{:02}'.format(i)] = defaultdict(dict)
        anno_set_path = os.path.join(dir_path, 'annotations',
                                     'set{:02}'.format(i))
        assert os.path.exists(anno_set_path), \
            'Not exists: '.format(anno_set_path)
        print('Extracting annotations from set{:02} ...'.format(i))
        for j in sorted(os.listdir(anno_set_path)):
            anno_path = os.path.join(anno_set_path, j)
            anno['{:02}'.format(i)][j[2:4]] = read_vbb(anno_path)

    with open(anno_save_path, 'w') as f:
        json.dump(anno, f)

    print('Annotations have been saved.')

    #  convert .seq file into .jpg
    for i in range(train_range[0], train_range[1]):
        convert_seq(i, dir_path, train_img_save_path, anno, interval=args.train_interval)
    for i in range(test_range[0], test_range[1]):
        convert_seq(i, dir_path, test_img_save_path, anno, interval=args.test_interval)

    print('Images have been saved.')

    print('Done, time spends: {}s'.format(int(time.time() - time_flag)))

if __name__ == '__main__':
    main()
