from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import os.path
import os
from itertools import chain
import ntpath
import multiprocessing as mp
from skimage.color import rgb2lab
from glob import glob
from math import cos, sin

def get_image_palette(img, nclusters):
    """
    Extract tuple of (Image, Palette) in LAB space
    """
    lab = rgb2lab(np.array(img))
    palette = kmeans_get_palette(lab, nclusters)
    return lab, palette

def get_augmented_image_palette(img, nclusters, angle):
    """
    Return tuple of (Image, Palette) in LAB space
    color shifted by the angle parameter
    """
    lab = rgb2lab(img)
    ch_a = lab[...,1]
    ch_b = lab[...,2]
    theta = np.deg2rad(angle)
    rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    hue_rotate = lambda ab: np.dot(rot, [ab[0], ab[1]])
    ab = np.asarray(list(map(hue_rotate, zip(ch_a, ch_b)))).transpose((0, 2, 1))
    lab = np.dstack((lab[...,0], ab[...,0], ab[...,1]))
    palette = kmeans_get_palette(lab, nclusters)
    return (lab, palette)


def kmeans_get_palette(img, nclusters):
    """
    Extract color palette of nclusters x 3 rgb values
    by performing a k-means clustering of the input image
    """
    h, w, c = img.shape
    km_im = np.array(img, dtype=np.float64).reshape(w * h, c)
    image_array_sample = shuffle(km_im, random_state=0, n_samples=1_000)
    kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(image_array_sample)
    codebook = kmeans.cluster_centers_
    return codebook

def save_image_palette(image, palette, path_base, angle=0):
    """
    Save preprocessed image and palette as .npz
    :param nd array image: Input image as nd array
    :param nd array palette: Input palette as nd array
    :param str path_base: Base save path
    :param int angle: Augmentation angle to use as a part of the save path
    """
    if(angle != 0):
        palette_file = os.path.join(path_base, str(angle).rjust(3, '0') + "_pal.npz")
        image_file = os.path.join(path_base, str(angle).rjust(3, '0') + "_img.npz")
    else:
        palette_file = os.path.join(path_base, "pal.npz")
        image_file = os.path.join(path_base, "img.npz")

    if(not os.path.exists(path_base)):
        os.makedirs(path_base)
    np.savez_compressed(image_file, image)
    np.savez_compressed(palette_file, palette)

def center_crop(img, w_target, h_target):
    """
    Center-crop a PIL image according to the 
    target dimensions
    """
    w_im, h_im = img.size
    # preconditions:
    if(w_target > w_im or h_target > h_im):
        return None
    # get the min dimension:
    min_dim = min(w_im, h_im)
    if w_im == min_dim:
        h_new = float(w_target) / float(w_im) * float(h_im)
        img = img.resize((int(w_target), int(h_new)))
        # crop height:
        d = int(0.5 * float(h_new - h_target))
        top = d
        bottom = d + h_target
        img = img.crop((0, top, w_target, bottom))
    else:
        w_new = float(h_target) / float(h_im) * float(w_im)
        img = img.resize((int(w_new), int(h_target)))
        # crop width:
        d = int(0.5 * float(w_new - w_target))
        left = d
        right = d + w_target
        img = img.crop((left, 0, right, h_target))
    return img
        

def preprocess_single(args):
    """
    Pre-process a single raw input image:
    1. Load input image
    2. Center crop to match training set dims
    3. Create a color-augmented version of the input by 40
       rotations in LAB color space
    4. Save the input and output image-palette pairs
    """
    im_path, source_path, target_path, w, h, n_p = args
    filename = os.path.splitext(ntpath.basename(im_path))[0]
    print('input: ' + im_path + '...')
    im = Image.open(im_path)
    w_im, h_im = im.size
    if(w_im < w or h_im < h):
        return

    angles = range(40, 360, 40)
    im_crop = center_crop(im, w, h)
    if im_crop == None:
        return
    im_crop = np.array(im_crop) / 255.
    aug_pairs = map(lambda angle: 
        get_augmented_image_palette(im_crop, n_p, angle), angles)
    # save image palette pairs (target)
    for (im_aug, pal), ang in zip(aug_pairs, angles):
        save_image_palette(im_aug, pal, os.path.join(target_path, filename), ang)
    # save source image palette pair
    im, palette = get_image_palette(im_crop, n_p)
    save_image_palette(im, palette, os.path.join(source_path, filename))


def preprocess(dataset_path, source_path, target_path, w, h, n_p, num_workers):
    """
    Preprocess raw input data in parallel
    """
    im_paths_png = (chain.from_iterable(glob(os.path.join(x[0], '*.png')) for x in os.walk(dataset_path)))
    im_paths_jpg = (chain.from_iterable(glob(os.path.join(x[0], '*.jpg')) for x in os.walk(dataset_path)))
    im_paths = [*im_paths_png, *im_paths_jpg]
    pool = mp.Pool(num_workers)
    pool.map(preprocess_single, [(im_path, source_path, target_path, w, h, n_p) for im_path in im_paths])
    pool.close()


if __name__ == '__main__':
    w = 256
    h = 256
    n_p = 6
    preprocess("./dataset/div2k", 
        source_path="./dataset/source", 
        target_path="./dataset/target", 
        w=w, 
        h=h, 
        n_p=6,
        num_workers=16
        )