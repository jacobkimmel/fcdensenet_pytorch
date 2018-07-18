'''
Generate a DataLoader for Lifeact Image Data
'''
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage.io import imread, imsave
from scipy.misc import imresize
from skimage.transform import resize
#imresize = resize # alias old scipy.misc.imresize
import glob
import os
from functools import partial
from torchvision import transforms, utils
import copy
from PIL import Image


def resize_sample(sample, size=(512,512,1)):
    '''
    Resizes input images and masks

    Parameters
    ----------
    sample : dict, {image, mask}.

    Returns
    -------
    sample_resized.
    '''

    d = {}
    for idx, key in enumerate(sample):
        imgR = imresize(sample[key], size)
        d[key] = imgR
    return d


class CellDataset(Dataset):
    '''Cell Segmentation Dataset'''

    def __init__(self, img_dir, mask_dir, transform=None, dtype='uint16', symlinks: bool=False):
        '''
        Parameters
        ----------
        img_dir : string. path to directory containing input images.
        mask_dir : string. path to directory containing ground truth masks.
        self.transform : callable. Optional transformer for samples.
        '''

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.imgs = glob.glob(os.path.join(img_dir, '*'))
        self.masks = glob.glob(os.path.join(mask_dir, '*'))
        self.imgs.sort()
        self.masks.sort()
        self.dtype = dtype
        self.symlinks = symlinks
        
        if self.symlinks:
            self.imgs = [os.path.realpath(x) for x in self.imgs]
            self.masks = [os.path.realpath(x) for x in self.masks]

        assert len(self.imgs) == len(self.masks),'Mismatched img and mask numbers'

    def __len__(self):
        return len(self.imgs)

    def _imload(self, imp):
        '''Load images using PIL or skimage.io'''
        if imp[-4:] == '.tif':
            image = np.array(Image.open(imp))
        else:
            image = imread(imp)
        return image

    def __getitem__(self, idx):
        image = self._imload(self.imgs[idx])
        mask = self._imload(self.masks[idx])
        # mask may be uint16 if not preprocessed with ignore_index labels
        if mask.dtype == 'uint16':
            # convert to set of unique values
            uniques = np.unique(mask)
            for u in range(len(uniques)):
                mask[mask == uniques[u]] == u
            mask = mask.astype('uint8')

        # expand dimensions if necessary
        if len(image.shape) < 3:
            image = np.expand_dims(image, -1)
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, -1)

        sample = {'image':image.astype(self.dtype), 'mask':mask.astype(self.dtype)}

        if self.transform:
            sample = self.transform(sample)

        return sample

class PredLoader(Dataset):
    '''Load only input images for predictions, pass blank masks to
    transforms in the sample'''

    def __init__(self, img_dir, transform=None, dtype='uint16', img_regex='*'):
        self.img_dir = img_dir
        self.transform = transform
        self.img_regex = img_regex
        self.imgs = glob.glob(os.path.join(img_dir, self.img_regex))
        self.imgs.sort()
        self.dtype = dtype

    def __len__(self):
        return len(self.imgs)

    def _imload(self, imp):
        '''Load images using PIL or skimage.io'''
        if imp[-4:] == '.tif':
            image = np.array(Image.open(imp))
        else:
            image = imread(imp)
        return image

    def __getitem__(self, idx):
        image = self._imload(self.imgs[idx])
        mask = np.zeros(image.shape).astype('uint8')
        if len(image.shape) < 3:
            image = np.expand_dims(image, -1)
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, -1)
        
        sample = {'image':image, 'mask':mask}
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToRGB(object):
    '''Converts 1-channel grayscale images to RGB'''

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = np.stack([image]*3, -1)
        image = np.squeeze(image)

        sample = {'image':image, 'mask':mask}
        return sample

class ToTensor(object):
    '''Convert ndarrays in sample to Tensors'''

    def __init__(self, type='float'):
        self.type = type

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype('float64')
        mask = mask.transpose((2, 0, 1)).astype('float64')
        if self.type == 'float':
            return {'image': torch.from_numpy(image).float(),
                    'mask': torch.from_numpy(mask).long()}
        elif self.type == 'byte':
            return {'image': torch.from_numpy(image).byte(),
                    'mask': torch.from_numpy(mask).long()}

class BinarizeMask(object):
    '''Binarizes masks'''

    def __init__(self):
        return

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        mask = mask.astype('bool')
        sample = {'image':image, 'mask':mask}
        return sample

        
class ChangeLabels(object):
    '''Changes a mask label to another in arrays'''

    def __init__(self, prev_label=2, new_label=-1):
        self.prev_label = prev_label
        self.new_label = new_label

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        mask[mask==self.prev_label] = self.new_label
        sample = {'image':image, 'mask':mask}
        return sample


class RescaleUnit(object):
    '''Rescales images to unit range [0,1]'''

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = image - image.min() # set min = 0
        image = image / image.max() # max / max = 1
        sample = {'image':image, 'mask':mask}
        return sample

class RandomFlip(object):
    '''Randomly flips image arrays'''

    def __init__(self, horz=True, vert=True, p=0.5):
        self.horz = horz
        self.vert = vert
        self.p = p

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if self.horz and np.random.random() > self.p:
            image = image[:,::-1,...]
            mask = mask[:,::-1,...]

        if self.vert and np.random.random() > self.p:
            image = image[::-1,:,...]
            mask = mask[::-1,:,...]

        sample = {'image':image, 'mask':mask}
        return sample

class SamplewiseCenter(object):
    '''Sets images to have 0 mean'''

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = image - image.mean()
        sample = {'image':image, 'mask':mask}
        return sample

class Resize(object):
    '''Resizes images'''

    def __init__(self, size=(512, 512, 1)):
        self.sz = size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        if len(image.shape) == 2:
            imageR = imresize(np.squeeze(image), self.sz)
        else:
            chans = []
            for c in range(image.shape[-1]):
                chanR = imresize(np.squeeze(image[...,c]), self.sz)
                chans.append(chanR)
            imageR = np.squeeze(np.stack(chans, axis=-1))
            
        mask = mask.astype('uint8')
        maskR = imresize(np.squeeze(mask), self.sz, interp='nearest')
        
        # reset mask to single integer labels
        for i in range(len(np.unique(maskR))):
            maskR[maskR == np.unique(maskR)[i]] = i

        if len(imageR.shape) < 3:
            imageR = np.expand_dims(imageR, -1)
        if len(maskR.shape) < 3:
            maskR = np.expand_dims(maskR, -1)


        return {'image':imageR, 'mask':maskR}


class ClassBalance(object):
    '''
    Class balances masks dynamically for semantic segmentation data
    using an `ignore_label`.
    Majority class representative pixels will be randomly selected at each
    pass over the image.

    Parameters
    ----------
    ignore_label : integer. label to ignored in loss calculations.
    p : float, [0,1]. proportion of not-ignored pixels from minority class.
    '''

    def __init__(self, ignore_index=-1, p=1):
        self.ignore_index = ignore_index
        self.p = p

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        mask = self.balance_mask(np.squeeze(mask), self.ignore_index, self.p)
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, -1)
        return {'image':image,'mask':mask}

    def balance_mask(self, mask, ignore_label, p):
        '''
        Generates a class balanced mask by marking pixels with a label to ignore
        Parameters
        ----------
        mask : ndarray.
        ignore_label : integer. Label to be ignored.
        p : float. ratio of total samples from minority class.
        '''
        mask = mask.astype('bool')
        nb_total = mask.shape[0]*mask.shape[1]

        err_mask = np.logical_not(mask)

        nb_class1 = mask.sum()
        if nb_class1 > 0.5*nb_total:
            return np.ones(mask.shape).astype('uint8')*np.int(ignore_label)

        nb_class0 = err_mask.sum()
        nb_class0_keep = int(nb_class1)*int(1/p)

        row, col = np.where(err_mask == True)
        idx = np.int32( np.random.choice(np.arange(len(row)), size=len(row), replace=False) )
        row_r = row[idx]
        col_r = col[idx]
        err_mask[row_r[:nb_class0_keep], col_r[:nb_class0_keep]] = False # all True in err_mask

        # set True values in error mask to -1 in y to encode error mask into y
        mask = np.clip(mask.astype('uint8'), 0, 1)
        mask[err_mask] = ignore_label
        return mask

class ClassBalanceMulti(object):
    '''
    Class balances masks that contain categorical classes.

    Parameters
    ----------
    ignore_lanel : integer. Label to be ignored.
    bal_label : integer. Label to balance against.
    p : ndarray. (n_classes) x 1 ratio of majority classes to minority class.
        i.e. p=[40,2,1] --> 40X, 2X more majority classes than minority class.
    priority_label : integer. label to prioritize at the same
        level as `bal_label` when class balancing.
    neg_label : integer. negative class label index. reduces multiclass balancing to 2 class problem
        with priority retention when combined with `priority_label`.
    '''

    def __init__( self, ignore_label=3, bal_label=1, p=np.array([1,1,1]), priority_label=None, neg_label=0., final_ignore_label=2. ):
        self.ignore_label = ignore_label
        self.bal_label = bal_label
        self.p = p
        self.priority_label = priority_label
        self.neg_label = neg_label
        self.final_ignore_label = final_ignore_label
        return

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if np.any(self.priority_label):
            mask = self.balance_mask_multi_duo(np.squeeze(mask))
        else:
            mask = self.balance_mask_multi(np.squeeze(mask), self.ignore_label, self.bal_label, self.p)
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, -1)
        return {'image':image,'mask':mask}

    def balance_mask_multi(self, mask, ignore_label, bal_label, p):
        n_classes = p.shape[0]
        class_counts = np.zeros(n_classes)
        for i in range(n_classes):
            class_counts[i] = (mask == i).sum()

        min_class = np.argmin(class_counts)
        maj_classes = [i for i in range(n_classes) if i != min_class]
        class_ratios = class_counts / (class_counts[bal_label]+1e-2) # avoid div by 0

        prop_class_keep = (np.ones(n_classes)*p)/(class_ratios+1e-2)

        new_mask = mask.copy()
        for c in range(n_classes):
            total = class_counts[c]
            keep = np.floor(total*prop_class_keep[c]).astype('int')
            off = np.max([total-keep, 0])

            if total==0 or off==0:
                continue

            rows, cols = np.where(new_mask==c)
            idx = np.arange(total).astype('int')
            # `ridx` are pixels above the keep rate to turn off
            ridx = np.random.choice(idx, size=np.int(off), replace=False)
            ridx = ridx.astype('int')
            new_mask[rows[ridx].astype('int'), cols[ridx].astype('int')] = ignore_label # turn off `ridx` pixels

        return new_mask

    def balance_mask_multi_duo(self, mask):
        n_classes = self.p.shape[0]
        class_counts = np.zeros(n_classes)
        for i in range(n_classes):
            class_counts[i] = (mask == i).sum()

        min_class = np.argmin(class_counts)
        maj_classes = [i for i in range(n_classes) if i != min_class]
        class_ratios = class_counts / (class_counts[self.bal_label]+1e-2) # avoid div by 0

        prop_class_keep = (np.ones(n_classes)*self.p)/(class_ratios+1e-2)

        priority_keep = np.min([class_counts[self.bal_label], class_counts[self.priority_label]])

        new_mask = mask.copy()
        for c in range(n_classes):
            total = class_counts[c]
            if c == self.priority_label:
                keep = priority_keep
            elif c == self.bal_label:
                keep = total
            else:
                keep_solo = np.floor(total*prop_class_keep[c]).astype('int')
                keep = keep_solo - priority_keep//(n_classes-2)

            off = np.max([total-keep, 0])

            if total==0 or off==0:
                continue

            rows, cols = np.where(new_mask==c)
            idx = np.arange(total).astype('int')
            # `ridx` are pixels above the keep rate to turn off
            ridx = np.random.choice(idx, size=np.int(off), replace=False)
            ridx = ridx.astype('int')
            new_mask[rows[ridx].astype('int'), cols[ridx].astype('int')] = self.ignore_label # turn off `ridx` pixels


        new_mask[new_mask == self.priority_label] = self.neg_label
        if self.final_ignore_label:
            new_mask[new_mask == self.ignore_label] = self.final_ignore_label
        return new_mask


# Transformer Zoo

grayscale_1024 = transforms.Compose([Resize(size=(1024,1024,1)), RescaleUnit(), SamplewiseCenter(), RandomFlip(), ClassBalance(ignore_index=2, p=0.02), ToTensor()])
grayscale_1024val = transforms.Compose([Resize(size=(1024,1024,1)), RescaleUnit(), SamplewiseCenter(), ToTensor()])
grayscale_1024nobal = transforms.Compose([Resize(size=(1024,1024,1)), RescaleUnit(), SamplewiseCenter(), RandomFlip(), ToTensor()])

grayscale_512 = transforms.Compose([Resize(size=(512,512,1)), RescaleUnit(), SamplewiseCenter(), RandomFlip(), ClassBalance(ignore_index=2, p=0.02), ToTensor()])
grayscale_512val = transforms.Compose([Resize(size=(512,512,1)), RescaleUnit(), SamplewiseCenter(), ToTensor()])
grayscale_512nobal = transforms.Compose([Resize(size=(512,512,1)), RescaleUnit(), SamplewiseCenter(), RandomFlip(), ToTensor()])

grayscale_512RGBnobal = transforms.Compose([Resize(size=(512,512,1)), RescaleUnit(), SamplewiseCenter(), RandomFlip(), ToRGB(), ToTensor()])
grayscale_512RGBval = transforms.Compose([Resize(size=(512,512,1)), RescaleUnit(), SamplewiseCenter(), ToRGB(), ToTensor()])

grayscale_1024RGBnobal = transforms.Compose([Resize(size=(1024,1024,1)), RescaleUnit(), SamplewiseCenter(), RandomFlip(),  ToRGB(), ToTensor()])
grayscale_1024RGB = transforms.Compose([Resize(size=(1024,1024,1)), RescaleUnit(), SamplewiseCenter(), RandomFlip(), ClassBalance(ignore_index=2, p=0.02), ToRGB(), ToTensor()])
grayscale_1024RGBval = transforms.Compose([Resize(size=(1024,1024,1)), RescaleUnit(), SamplewiseCenter(), ToRGB(), ToTensor()])

grayscale_128 = transforms.Compose([Resize(size=(128,128,1)), RescaleUnit(), SamplewiseCenter(), RandomFlip(), ClassBalance(ignore_index=2, p=0.02), ToTensor()])
grayscale_128val = transforms.Compose([Resize(size=(128,128,1)), RescaleUnit(), SamplewiseCenter(), ToTensor()])
grayscale_128nobal = transforms.Compose([Resize(size=(128,128,1)), RescaleUnit(), SamplewiseCenter(), RandomFlip(), ToTensor()])

basic_512 = transforms.Compose([BinarizeMask(),
                                Resize(size=(512,512,1)), 
                                RescaleUnit(), 
                                SamplewiseCenter(), 
                                RandomFlip(), 
                                ToTensor()])

basic_256 = transforms.Compose([Resize(size=(256,256,1)), 
                                RescaleUnit(), 
                                SamplewiseCenter(), 
                                RandomFlip(), 
                                ToTensor()])

basic_256v = transforms.Compose([Resize(size=(256,256,1)), 
                                RescaleUnit(), 
                                SamplewiseCenter(), 
                                ToTensor()])