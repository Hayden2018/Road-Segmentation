import os
import pickle as pk
from random import randint

import cv2 as cv
import numpy as np
import torch
from matplotlib import pyplot as plt


# create a pickle file of the 3d np.array containing the image of whole target region
def create_map(pos='/z20_sat_boston', name='sat_data_boston'):
    cwd = os.getcwd()
    for current, directories, files in os.walk(cwd + pos):
        os.chdir(current)
        try:
            vertical = cv.imread(files[0], cv.IMREAD_COLOR)
            i = 1
            while i < len(files):
                tmp = cv.imread(files[i], cv.IMREAD_COLOR)
                vertical = np.concatenate((vertical, tmp), axis=0)
                i = i + 1
            try:
                map = np.concatenate((map, vertical), axis=1)
            except:
                map = vertical
        except:
            pass
    os.chdir(cwd)
    f = open(name, 'wb')
    pk.dump(map, f)


# show a image in the form of 3d or 2d np.array
def show(img):
    try:
        b, g, r = cv.split(img)
        img = cv.merge([r, g, b])
        plt.imshow(img)
        plt.show()
    except:
        plt.imshow(img)
        plt.show()


# (2d or 3d np.array) return array of of slices
def cut_image(img, slice=64):
    if isinstance(img, np.ndarray):
        pass
    else:
        img = np.array(img)
    tmp = np.vsplit(img, slice)
    output = np.hsplit(tmp[0], slice)
    output = np.array(output)
    j = 1
    while j < len(tmp):
        x = np.hsplit(tmp[j], slice)
        x = np.array(x)
        output = np.concatenate((output, x), axis=0)
        j += 1
    return output


# return the mode of 1d python list
def mode(x):
    dict = {}
    for i in x:
        if i in dict:
            dict[i] = dict[i] + 1
        else:
            dict[i] = 1
    max = 0
    for i in dict:
        if dict[i] > max:
            max = dict[i]
            output = i
    return output


# (np.array, 2d list, tile = int, size = int) return 3d np.array, 3d np.array
def overlay(img, labels, tile=64, size=4):
    color = [[255, 255, 255], [0, 0, 255], [0, 255, 0], [255, 0, 0], [128, 0, 128]]
    y = []
    i = 0
    while i < tile:
        tmp = []
        j = 0
        while j < tile:
            tmp.append(color[labels[i][j]])
            j = j + 1
        y.append(tmp)
        i = i + 1
    layer = np.array(y, dtype=np.uint8)
    layer = cv.resize(layer, None, fx=size, fy=size, interpolation=cv.INTER_AREA)
    output = cv.addWeighted(img, 0.7, layer, 0.2, 0)
    return output, layer


# (2d list, size = int) return 2d python list
def mode_kernel(x, size=4):
    x = np.array(x)
    i, j = 0, 0
    y = []
    while i < x.shape[0]:
        tmp = []
        j = 0
        while j < x.shape[1]:
            window = x[i:i + size, j:j + size]
            window = window.flatten()
            window = window.tolist()
            m = mode(window)
            tmp.append(m)
            j = j + size
        y.append(tmp)
        i = i + size
    return y


# (2d list, absorb_margin = int) return 2d python list
def absorption(y, absorb_margin=2):
    z = []
    i = 1
    while i < len(y) - 1:
        j = 1
        tmp = []
        while j < len(y[0]) - 1:
            count = 0
            window = [y[i - 1][j - 1]]
            window.append(y[i][j - 1])
            window.append(y[i + 1][j - 1])
            window.append(y[i - 1][j])
            window.append(y[i + 1][j])
            window.append(y[i - 1][j + 1])
            window.append(y[i][j + 1])
            window.append(y[i + 1][j + 1])
            for k in window:
                if k == y[i][j]: count = count + 1
            if count > absorb_margin:
                m = y[i][j]
            else:
                m = mode(window)
            tmp.append(m)
            j = j + 1
        z.append(tmp)
        i = i + 1
    y = np.array(y, dtype=np.uint8)
    z = np.array(z, dtype=np.uint8)
    y[1:y.shape[0] - 1, 1:y.shape[1] - 1] = z
    y = y.tolist()
    return y


def random_noise(img):
    x = np.random.randint(0, 200, size=img.shape, dtype=np.uint8)
    img = cv.addWeighted(img, 0.9, x, 0.1, 0)
    return img


# Direct label roads at pixel level based on Google map color.
def direct_label(img):
    t1 = [196, 231, 253]
    t2 = [255, 255, 255]
    t3 = [157, 216, 254]
    output = []
    img = img.tolist()
    for i in img:
        output.append([])
        for j in i:
            x = j[0] == t1[0] and j[1] == t1[1] and j[2] == t1[2]
            y = j[0] == t2[0] and j[1] == t2[1] and j[2] == t2[2]
            z = j[0] == t3[0] and j[1] == t3[1] and j[2] == t3[2]
            if x or y or z:
                output[-1].append(0)
            else:
                output[-1].append(randint(1, 3))
    output = np.array(output, dtype=np.uint8)
    return output


# Flatten 2d python list
def flatten(l):
    output = []
    for i in l:
        for j in i:
            output.append(j)
    return output


# Set all non-zero element in a 2d list to 1
def purify(x):
    for i in x:
        for j in range(len(i)):
            if i[j] != 0: i[j] = 1
    return x


def ToTensor(x, type=0):
    x = torch.Tensor(x)
    if type == 'image':
        if len(x.shape) == 3:
            x = x.permute(2, 0, 1)
        else:
            x = x.permute(0, 3, 1, 2)
    if type == 'label':
        x = x.type(torch.long)
    return x


def ToArray(x, type=0):
    x = x.cpu().detach()
    if type == 'image':
        x = x.type(torch.uint8)
        if len(x.shape) == 3:
            pass
        else:
            x = x.permute(0, 2, 3, 1)
    if type == 'label':
        x = x.argmax(dim=-3)
    x = x.numpy()
    return x


def sequential_iterator(step=10, iteration=10, high=1000):
    j = 0
    while j < high - 1:
        for k in range(iteration):
            yield j
        j += step


def random_iterator(cycle=60, iteration=10, high=180):
    for i in range(cycle):
        r = randint(0, high)
        for j in range(iteration):
            yield r


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h))
    return rotated


if __name__ == '__main__':
    with open('map_data_boston', 'rb') as f:
        mp = pk.load(f)
        mp = mp[20000:22048, 20000:22048]
    show(mp)
    labels = direct_label(mp)
    del mp
    with open('sat_data_boston', 'rb') as f:
        sample = pk.load(f)
        sample = sample[20000:22048, 20000:22048]
    labels = mode_kernel(labels, size=4)
    labels = absorption(labels, absorb_margin=3)
    labels = absorption(labels, absorb_margin=3)
    labels = purify(labels)
    labels = np.array(labels)
    show(labels)
    sample = cut_image(sample, slice=4)
    labels = cut_image(labels, slice=4)
    data = []
    tmp = []
    for i in range(16):
        s = np.sum(labels[i])
        if s < 15000:
            data.append(sample[i])
            tmp.append(labels[i])
    labels = np.array(tmp)
    data = np.array(data)
    print('There are', data.shape[0], 'data.')
    with open('nn_data_test', 'wb') as f:
        pk.dump(data, f)
    with open('nn_labels_test', 'wb') as f:
        pk.dump(labels, f)
