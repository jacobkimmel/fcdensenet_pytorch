'''
Generate a DataLoader for Cell Image Data
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
import tifffile

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

    def __init__(self, 
                 img_dir: str,
                 mask_dir: str, 
                 transform=None, 
                 dtype: str='uint16', 
                 symlinks: bool=False, 
                 name_check: bool=False,
                 samples_per_image: int=1):
        '''
        Parameters
        ----------
        img_dir : str
            path to directory containing input images.
        mask_dir : str
            path to directory containing ground truth masks.
        transform : Callable
            Optional transformer for samples.
        dtype : str
            input data type using numpy syntax.
        symlinks : bool
            inputs are symlink paths.
        name_check : bool
            perform a check to ensure names match a format of 
            "img_name.EXT" <> "img_name_mask.EXT"
        samples_per_image : int
            number of samples to yield per image.
            set to 1 unless performing a transform that randomizes input
            images in some dramatic way (i.e. cropping subwindows).
        '''

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.imgs = sorted(glob.glob(os.path.join(img_dir, '*')))
        self.masks = sorted(glob.glob(os.path.join(mask_dir, '*')))
        self.dtype = dtype
        self.symlinks = symlinks
        self.samples_per_image = samples_per_image

        if self.symlinks:
            self.imgs = [os.path.realpath(x) for x in self.imgs]
            self.masks = [os.path.realpath(x) for x in self.masks]

        assert len(self.imgs) == len(self.masks),'Mismatched img and mask numbers'
        
        if name_check:
            img_names = [os.path.splitext(os.path.basename(x))[0] for x in self.imgs]
            mask_names = [os.path.basename(x).split('_mask')[0] for x in self.masks]
            same = [True if img_names[i] == mask_names[i] else False for i in range(len(img_names))]
            assert np.all(same)

    def __len__(self):
        return int(len(self.imgs)*self.samples_per_image)

    def _imload(self, imp):
        '''Load images using tifffile or PIL'''
        if imp[-4:] == '.tif':
            image = tifffile.TiffFile(imp).asarray()
        else:
            image = np.array(Image.open(imp))
        return image

    def __getitem__(self, idx):
        '''Fetch a sample'''
        if self.samples_per_image > 1:
            # set to appropriate place in the index
            idx = idx % len(imgs)
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
        '''Load images using tifffile or PIL'''
        if imp[-4:] == '.tif':
            image = tifffile.TiffFile(imp).asarray()
        else:
            image = np.array(Image.open(imp))
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


class CellCropDataset(Dataset):
    '''Load input images for prediction, performing structured crops to
    provide a consistent testing set.
    '''

    def __init__(self,
                 img_dir,
                 mask_dir,
                 transform_pre=None,
                 transform_post=None,
                 dtype='uint16',
                 img_regex='*',
                 mask_regex='*',
                 n_windows=4,
                 symlinks=True):
        '''
        Parameters
        ----------
        img_dir : str
            path to image inputs
        transform_pre : callable
            transform to apply prior to cutting images into windows.
        transform_post : callable
            transform to apply after cutting images into windows.
        dtype : str
            dtype of images
        img_regex : str
            pattern to glob image filenames
        n_windows : int
            number of panels to split the pre-split transformed
            image into. Must be a perfect square.
        '''

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform_pre = transform_pre
        self.transform_post = transform_post
        self.img_regex = img_regex
        self.mask_regex = mask_regex
        self.imgs  = sorted(glob.glob(os.path.join(img_dir, self.img_regex)))
        self.masks = sorted(glob.glob(os.path.join(mask_dir, self.mask_regex)))
        self.dtype = dtype
        self.n_windows = n_windows

        self.symlinks = symlinks

        if self.symlinks:
            self.imgs = [os.path.realpath(x) for x in self.imgs]
            self.masks = [os.path.realpath(x) for x in self.masks]

    def __len__(self):
        '''Each window is a sample'''
        return len(self.imgs) * self.n_windows

    def _imload(self, imp):
        '''Load images using tifffile or PIL'''
        if imp[-4:] == '.tif':
            image = tifffile.TiffFile(imp).asarray()
        else:
            image = np.array(Image.open(imp))
        return image

    def _split_windows(self, sample):
        '''
        Parameters
        ----------
        sample : dict
            keys `image` and `mask`, np.ndarrays of [H, W, C]

        Returns
        -------
        windows : list
            np.ndarrays of [H, W, C] of image windows in the order
            of left to right, top to bottom.
        '''
        image, mask = sample['image'], sample['mask']
        total_shape = np.array(image.shape)[:2]

        n_per_side = int(np.sqrt(self.n_windows))
        sz_per_side = total_shape // n_per_side
        h_sz, w_sz = sz_per_side
        image_windows = []
        mask_windows = []
        # perform a right to left, top to bottom raster of windows
        for i in range(self.n_windows):
            iw = image[(i//n_per_side)*h_sz:((i//n_per_side)+1)*h_sz,
                       (i%n_per_side)*h_sz:((i%n_per_side)+1)*h_sz]
            im = mask[(i//n_per_side)*h_sz:((i//n_per_side)+1)*h_sz,
                      (i%n_per_side)*h_sz:((i%n_per_side)+1)*h_sz]
            image_windows.append(iw)
            mask_windows.append(im)
        return image_windows, mask_windows

    def __getitem__(self, idx):
        '''
        Parameters
        ----------
        idx : int
            subimage ("window") to load.

        Returns
        -------
        samples : list
            contains `self.n_windows` dicts, each keyed `image`
            and `mask` for processing.
            see `._split_windows` for ordering.
        '''
        iidx = idx // self.n_windows
        image = self._imload(self.imgs[iidx])
        mask = self._imload(self.masks[iidx])

        # mask may be uint16 if not preprocessed with ignore_index labels
        if mask.dtype == 'uint16':
            # convert to set of unique values
            uniques = np.unique(mask)
            for u in range(len(uniques)):
                mask[mask == uniques[u]] == u
            mask = mask.astype('uint8')

        if len(image.shape) < 3:
            image = np.expand_dims(image, -1)
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, -1)

        sample = {'image':image, 'mask':mask}

        if self.transform_pre:
            sample = self.transform_pre(sample)

        # break image into panels
        image_windows, mask_windows = self._split_windows(sample)

        samples = [{'image': image_windows[i],
                    'mask' : mask_windows[i]} for i in range(len(image_windows))]

        sidx = idx % self.n_windows
        sample = samples[sidx]
        if self.transform_post:
            sample = self.transform_post(sample)

        return sample

class PredCropLoader(Dataset):
    '''Load only input images for predictions and split each image into a panel
    of windows, size `crop`
    '''

    def __init__(self,
                 img_dir,
                 transform_pre=None,
                 transform_post=None,
                 dtype='uint16',
                 img_regex='*',
                 n_windows=4):
        '''
        Parameters
        ----------
        img_dir : str
            path to image inputs
        transform_pre : callable
            transform to apply prior to cutting images into windows.
        transform_post : callable
            transform to apply after cutting images into windows.
        dtype : str
            dtype of images
        img_regex : str
            pattern to glob image filenames
        n_windows : int
            number of panels to split the pre-split transformed
            image into. Must be a perfect square.
        '''

        self.img_dir = img_dir
        self.transform_pre = transform_pre
        self.transform_post = transform_post
        self.img_regex = img_regex
        self.imgs = glob.glob(os.path.join(img_dir, self.img_regex))
        self.imgs.sort()
        self.dtype = dtype
        self.n_windows = n_windows

    def __len__(self):
        return len(self.imgs)

    def _imload(self, imp):
        '''Load images using tifffile or PIL'''
        if imp[-4:] == '.tif':
            image = tifffile.TiffFile(imp).asarray()
        else:
            image = np.array(Image.open(imp))
        return image

    def _split_windows(self, sample):
        '''
        Parameters
        ----------
        sample : dict
            keys `image` and `mask`, np.ndarrays of [H, W, C]

        Returns
        -------
        windows : list
            np.ndarrays of [H, W, C] of image windows in the order
            of left to right, top to bottom.
        '''
        image = sample['image']
        total_shape = np.array(image.shape)[:2]

        n_per_side = int(np.sqrt(self.n_windows))
        sz_per_side = total_shape // n_per_side
        h_sz, w_sz = sz_per_side
        windows = []
        # perform a right to left, top to bottom raster of windows
        for i in range(self.n_windows):
            w = image[(i//n_per_side)*h_sz:((i//n_per_side)+1)*h_sz,
                      (i%n_per_side)*h_sz:((i%n_per_side)+1)*h_sz]
            windows.append(w)
        return windows

    def __getitem__(self, idx):
        '''
        Parameters
        ----------
        idx : int
            image to load.

        Returns
        -------
        samples : list
            contains `self.n_windows` dicts, each keyed `image`
            and `mask` for processing.
            see `._split_windows` for ordering.
        '''
        image = self._imload(self.imgs[idx])
        mask = np.zeros(image.shape).astype('uint8')
        if len(image.shape) < 3:
            image = np.expand_dims(image, -1)
        if len(mask.shape) < 3:
            mask = np.expand_dims(mask, -1)

        sample = {'image':image, 'mask':mask}

        if self.transform_pre:
            sample = self.transform_pre(sample)

        # break image into panels
        windows = self._split_windows(sample)

        samples = [{'image':windows[i], 'mask':np.zeros_like(windows[i])} for i in range(len(windows))]

        if self.transform_post:
            new_samples = []
            for i, s in enumerate(samples):
                s = self.transform_post(s)
                new_samples.append(s)
            samples = new_samples

        return samples

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

class RandomCrop(object):
    '''Randomly crops an image'''

    def __init__(self, crop_sz=(512, 512), min_mask_sum=0):
        self.crop_sz = np.array(crop_sz).astype('int')
        self.min_mask_sum = min_mask_sum

    def __call__(self, sample):
        '''
        sample : dict
            'image' : [H, W, C] np.ndarray
            'mask'  : [H, W] np.ndarray
        '''
        image, mask = sample['image'], sample['mask']

        max_hidx = image.shape[0] - self.crop_sz[0]
        max_widx = image.shape[1] - self.crop_sz[1]

        find_idx = True
        while find_idx:
            hidx = int(np.random.choice(np.arange(max_hidx), size=1).astype('int')[0])
            widx = int(np.random.choice(np.arange(max_widx), size=1).astype('int')[0])

            assert type(hidx) is int and type(widx) is int
            assert hidx+self.crop_sz[0] < image.shape[0]
            assert widx+self.crop_sz[1] < image.shape[1]

            imageC = image[hidx : hidx + self.crop_sz[0],
                       widx : widx + self.crop_sz[1],
                      :] # leave channels alone
            maskC  = mask[hidx : hidx + self.crop_sz[0],
                      widx : widx + self.crop_sz[1],
                      :] # leave channels alone
            if maskC.sum() >= self.min_mask_sum:
                find_idx = False
            if mask.sum() <= 2*self.min_mask_sum:
                find_idx = False

        sample = {'image':imageC, 'mask':maskC}
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

# Transformer Zoo

basic_512 = transforms.Compose([Resize(size=(512,512,1)),
                                RescaleUnit(),
                                SamplewiseCenter(),
                                RandomFlip(),
                                ToTensor()])

basic_512v = transforms.Compose([Resize(size=(512,512,1)), 
                                RescaleUnit(), 
                                SamplewiseCenter(), 
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



crop512 = transforms.Compose([Resize(size=(2048,2048,1)),
                               RandomCrop(crop_sz=(512,512), min_mask_sum=1),
                                RescaleUnit(),
                                SamplewiseCenter(),
                                RandomFlip(),
                                ToTensor()])

crop512raw = transforms.Compose([
                              BinarizeMask(),
                              RandomCrop(crop_sz=(512,512)),
                              RescaleUnit(),
                              SamplewiseCenter(),
                              RandomFlip(),
                              ToTensor()])

predcrop512_pre = transforms.Compose([Resize(size=(2048,2048,1))])
predcrop512 = transforms.Compose([
                                BinarizeMask(),
                                RescaleUnit(),
                                SamplewiseCenter(),
                                ToTensor()])
