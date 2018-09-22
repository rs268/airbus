import numpy as np
import pandas as pd
import os, os.path

#ref https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def decode(mask_string, shape=(768,768)):
    '''
    Decodes run length encoding of image mask and returns a numpy
    array of 0's and 1's, i.e. the mask of the image.
    '''

    if isinstance(mask_string, str):
        encoding = mask_string.split()

        start, length = [np.asarray(x, dtype=int) for x in (encoding[0:][::2], encoding[1:][::2])]

        start -= 1
        end = start + length

        mask = np.zeros(shape[0]*shape[1], dtype=np.int8)

        for l, h in zip(start, end):
            mask[l:h] = 1

        return mask.reshape(shape)

    else:
        return np.zeros(shape, dtype=np.int8)

'''
 Builds the Dataset object's train_df and test_df variables
'''
def build(train_path, test_path, seg_path):
    df = pd.read_csv(os.path.join(seg_path, 'train_ship_segmentations.csv'))

    train_imgs = pd.DataFrame(data=images(train_path), columns=['ImageId', 'Image'])
    test_imgs = pd.DataFrame(data=images(test_path), columns=['ImageId', 'Image'])

    train_df = train_imgs.merge(df, how='left', on='ImageId')
    test_df = test_imgs.merge(df, how='left', on='ImageId')

    return (train_df, test_df)

def images(path):
    files = os.listdir(path)
    paths = [os.path.join(path, x) for x in files]

    return {'ImageId':files, 'Image':paths}

'''
Converts an image path in to a numpy array and
to take an RLE encoded string and turn it in to a numpy array
'''
def convert(df):
    df['EncodedPixels'] = decode(df['EncodedPixels'])
    path = df['Image']

    img = get_image_arrays(path)

    df['Image'] = img
    return df

def get_image_arrays(path):
    img = Image.open(path)
    im = np.asarray(img)
    img.close()

    return im