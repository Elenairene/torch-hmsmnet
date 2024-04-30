import cv2
import random
import numpy as np
import scipy.signal as sig
from torch.utils.data import Dataset

kx = np.array([[-1, 0, 1]])
ky = np.array([[-1], [0], [1]])

def fill(img):
    masktrue = (img !=0)
    maskfalse = (img ==0)
    meanimg = np.mean(img[masktrue])
    newimg = img.copy()
    newimg[maskfalse] = meanimg

    return newimg
def read_left(filename, mean=None, std = None):
    filename = filename.replace("withgroundtruth", "whu")
    # print(filename)
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    meani, stdi = mean, std
    if (len(np.shape(img))>2):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img = fill(img)
    if meani == None:
        meani = np.mean(img)
    if stdi == None:
        stdi = np.std(img)
    img = (img - meani) / stdi
    # print(meani)
    dx = sig.convolve2d(img, kx, 'same')
    dy = sig.convolve2d(img, ky, 'same')
    img = np.expand_dims(img.astype('float32'), -1)
    dx = np.expand_dims(dx.astype('float32'), -1)
    dy = np.expand_dims(dy.astype('float32'), -1)
    return img, dx, dy


def read_right(filename, mean=None, std = None):
    filename = filename.replace("withgroundtruth", "whu")
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    #img = fill(img)
    meani, stdi = mean, std

    if (len(np.shape(img))>2):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    if meani == None:
        meani = np.mean(img)
    if stdi == None:
        stdi = np.std(img)
    img = (img - meani) / stdi
    return np.expand_dims(img.astype('float32'), -1)


def read_disp(filename):
    filename = filename.replace("withgroundtruth", "whu")
    disp = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    disp_16x = cv2.resize(disp, (64, 64)) / 16.0
    disp_8x = cv2.resize(disp, (128, 128)) / 8.0
    disp_4x = cv2.resize(disp, (256, 256)) / 4.0
    disp = np.expand_dims(disp, -1)
    disp_16x = np.expand_dims(disp_16x, -1)
    disp_8x = np.expand_dims(disp_8x, -1)
    disp_4x = np.expand_dims(disp_4x, -1)
    return disp_16x, disp_8x, disp_4x, disp


def read_batch(left_paths, right_paths, disp_paths):
    lefts, dxs, dys, rights, d16s, d8s, d4s, ds = [], [], [], [], [], [], [], []
    for left_path, right_path, disp_path in zip(left_paths, right_paths, disp_paths):
        left, dx, dy = read_left(left_path)
        right = read_right(right_path)
        d16, d8, d4, d = read_disp(disp_path)
        lefts.append(left)
        dxs.append(dx)
        dys.append(dy)
        rights.append(right)
        d16s.append(d16)
        d8s.append(d8)
        d4s.append(d4)
        ds.append(d)
    return np.array(lefts), np.array(rights), np.array(dxs), np.array(dys),\
           np.array(d16s), np.array(d8s), np.array(d4s), np.array(ds)


def load_batch(all_left_paths, all_right_paths, all_disp_paths, batch_size=4, reshuffle=False):
    assert len(all_left_paths) == len(all_disp_paths)
    assert len(all_right_paths) == len(all_disp_paths)

    i = 0
    while True:
        lefts, rights, dxs, dys, d16s, d8s, d4s, ds = read_batch(
            left_paths=all_left_paths[i * batch_size:(i + 1) * batch_size],
            right_paths=all_right_paths[i * batch_size:(i + 1) * batch_size],
            disp_paths=all_disp_paths[i * batch_size:(i + 1) * batch_size])
        yield [lefts, rights, dxs, dys], [d16s, d8s, d4s, ds]
        i = (i + 1) % (len(all_left_paths) // batch_size)
        if reshuffle:
            if i == 0:
                paths = list(zip(all_left_paths, all_right_paths, all_disp_paths))
                random.shuffle(paths)
                all_left_paths, all_right_paths, all_disp_paths = zip(*paths)
class CustomDataset(Dataset):
    def __init__(self, listfilename, training):
        self.training = training
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(listfilename)
    def load_path(self, listfilename):
        with open(listfilename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disparities = [x[2] for x in splits]
        #disparitiesr = [x[3] for x in splits]
        return left_images, right_images, disparities#, disparitiesr
    def __len__(self):
        return len(self.left_filenames)
    def read_left(self,left_dir):
        filename = left_dir#.replace("withgroundtruth", "whu")
        print(filename)
        # print(filename)
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if (len(np.shape(img))>2):
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #img = fill(img)
        meani = np.mean(img)
        stdi = np.std(img)
        img = (img - meani) / stdi
        # print(meani)
        dx = sig.convolve2d(img, kx, 'same')
        dy = sig.convolve2d(img, ky, 'same')
        # img = np.expand_dims(img.astype('float32'), -1)
        # dx = np.expand_dims(dx.astype('float32'), -1)
        # dy = np.expand_dims(dy.astype('float32'), -1)
        return img.astype('float32'), dx.astype('float32'), dy.astype('float32')
    def read_right(self,right_dir):
        filename = right_dir#.replace("withgroundtruth", "whu")
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        #img = fill(img)
        # meani, stdi = mean, std

        if (len(np.shape(img))>2):
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        meani = np.mean(img)
        stdi = np.std(img)
        img = (img - meani) / stdi
        return img.astype('float32') #np.expand_dims(img.astype('float32'), -1)
    def read_disp(self,dispname):
        filename = dispname#.replace("withgroundtruth", "whu")
        disp = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        # disp_16x = cv2.resize(disp, (64, 64)) / 16.0
        # disp_8x = cv2.resize(disp, (128, 128)) / 8.0
        # disp_4x = cv2.resize(disp, (256, 256)) / 4.0
        disp = np.expand_dims(disp, 0)
        # disp_16x = np.expand_dims(disp_16x, -1)
        # disp_8x = np.expand_dims(disp_8x, -1)
        # disp_4x = np.expand_dims(disp_4x, -1)
        return disp
    def __getitem__(self,index):
        # lefts, rights, disparitygt = self.readfromtxt(testlist) #disparityrgt
        left_image_ori, gx_ori, gy_ori = self.read_left(self.left_filenames[index])
        right_image_ori = self.read_right(self.right_filenames[index])
        disp = self.read_disp(self.disp_filenames[index])
        
        left_image_ori = np.expand_dims(left_image_ori, 0)
        gx_ori = np.expand_dims(gx_ori, 0)
        gy_ori = np.expand_dims(gy_ori, 0)
        right_image_ori = np.expand_dims(right_image_ori, 0)
        return {"left_filename":self.left_filenames[index],
            "left":left_image_ori, 
        "right":right_image_ori, 
        "gx":gx_ori, 
        "gy":gy_ori,
        "disp":disp}