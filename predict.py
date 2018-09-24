'''Run predictions'''
import sys
sys.path = sys.path + ['/home/jacob/src/fcdensenet_pytorch/']
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from model import DenseNet103
from data_loader import PredLoader, basic_256v
import os
import glob
import time
import argparse
from scipy.misc import imresize
from skimage.morphology import remove_small_objects, binary_erosion, binary_dilation, disk
from skimage.io import imsave

class Ensemble(object):
    def __init__(self, models):
        '''
        models : list. list of model objects with a callable .forward() method.
        '''
        self.models = models

    def __call__(self, inputs):
        '''Return the mean prediction of all models in `models`'''
        outs = []
        for m in self.models:
            outs.append(m(inputs))
        output = torch.stack(outs)
        output = output.sum(0)/len(outs)
        return output

def predict(model, inputs):
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    inputs.requires_grad = False
    out = model(inputs)

    return out


def post_process(sm_out, min_sm_prob=0.50, sz_min=2000, cell_idx=1, edge_idx=2, erod=False, dil=False):
    '''
    Post process a softmax output image.

    Parameters
    ----------
    sm_out : ndarray. 1 x C x H x W,
        where C is the number of classes.
        Softmax output, so sm_out.sum(1) ~= 1 at each pixel.
        May be off by a small error eta due to floating point rounding.
    min_sm_prob : float, [0, 1]. minimum softmax score to be
        considered an object.
    sz_min : float. minimum object area.
    cell_idx : int. class number for the cell interiors.
    edge_idx : int. class number for cell edges.
    erod : integer or False. Size of erosion disk selem.
    dil : integer or False. Size of dilation disk selem.

    Returns
    -------
    pp_mask : ndarray, H x W. binary.
        post processing binary segmentation mask.
    '''
    # Check that the output is a Softmax out
    softmax_check = (sm_out.sum(1) > 0.99).sum() == sm_out.shape[2]*sm_out.shape[3]
    assert softmax_check, 'Class scores do not sum to one. Is this softmax output?'

    mask = np.squeeze(sm_out[:,cell_idx,:,:] > min_sm_prob)
    mask[np.squeeze(sm_out[:,edge_idx,:,:] > min_sm_prob)] = 0 # zero borders
    mask_c = remove_small_objects(mask, sz_min)
    if erod:
        mask_f = binary_erosion(mask_c, disk(erod))
    if dil:
        mask_f = binary_dilation(mask_c, disk(dil))
    else:
        mask_f = mask_c
    return mask_f

def main():
    parser = argparse.ArgumentParser(description='Segment images using trained CellSegNet models')
    parser.add_argument('in_dir', type=str, help='image directory of input images')
    parser.add_argument('out_dir', type=str, help='output directory for segmented images')
    parser.add_argument('models', type=str, nargs='+', help='paths to model weights. multiple models will be Ensembled.')
    parser.add_argument('--img_regex', type=str, default='*', help='regular expression matching image filenames in `in_dir`')
    parser.add_argument('--n_classes', type=int, default=3, help='number of classes in the output masks')
    parser.add_argument('--min_smp', type=float, default=0.50, help='minimum softmax score for object pixels')
    parser.add_argument('--sz_min', type=float, default=10., help='minimum object size for post-processing')
    parser.add_argument('--erod', default=False, help='size of disk selem for post-processing erosion')
    parser.add_argument('--dil', default=False, help='size of disk selem for post-processing dilation')
    parser.add_argument('--upsamp_sz', type=int, nargs=2, default=[1420, 1040], help='final size of upsampled mask')
    parser.add_argument('--quiet', action='store_true', default=False, help='suppress verbose output')
    parser.add_argument('--batch', type=int, default=1, help='batch size for predictions')

    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    model_paths = args.models
    img_regex = args.img_regex
    min_smp = args.min_smp
    sz_min = args.sz_min
    erod = int(args.erod)
    dil = int(args.dil)
    upsamp_sz = tuple(args.upsamp_sz)
    verbose = np.logical_not(args.quiet)
    batch_size = int(args.batch)

    # Get image names
    img_files = glob.glob(os.path.join(in_dir, img_regex))
    img_files.sort()

    img_names = [x.split('/')[-1][:-4] for x in img_files] # names w/o extensions

    # Load models and build ensemble
    models = []
    for m in model_paths:
        mdl = DenseNet103(n_classes=args.n_classes)
        mdl.load_state_dict(torch.load(m))
        if torch.cuda.is_available():
            mdl = mdl.cuda()
        mdl.train(False)
        models.append(mdl)

    ens = Ensemble(models)
    print('Models loaded.')
    # Set up data loaders
    pl = PredLoader(in_dir, transform=basic_256v, dtype='uint16', img_regex=img_regex)
    print('Data loader initialized.')
    print(len(pl), ' images found.')
    # initialize softmax fnx
    sm = torch.nn.Softmax2d()


    dl = DataLoader(pl, batch_size=batch_size, num_workers=6, shuffle=False)
    iter_dl = iter(dl)
    # Segment images
    print('Segmenting...')
    times = []
    for i in range(len(dl)):
        start = time.time()
        batch = next(iter_dl)
        inputs = batch['image']
        s = pl[i]
        img = s['image']
        print(img.size())
        outs = predict(ens, inputs)
        probs = sm(outs)
        probs = probs.cpu().data.numpy() # unpack to numpy array

        for j in range(probs.shape[0]):
            mask = post_process(probs[j:j+1,...], min_sm_prob=min_smp, sz_min=sz_min, erod=erod, dil=dil)
            maskR = imresize(mask.astype('uint8'), upsamp_sz, interp='nearest') # upsample
            maskR = maskR.astype('bool').astype('uint8')
            # save upsamples mask
            imsave(os.path.join(out_dir, img_names[i*batch_size + j] + '_csn.png'), maskR*255)
            if verbose:
                print('Processed ', img_names[i*batch_size + j])
        end = time.time()
        print('Batch in ', end-start, 'seconds')
        print('Image mean ', (end-start)/batch_size, 'seconds')
        times.append(end-start)

    print('Average image processing time : ', np.mean(times))


    return

if __name__ == '__main__':
    main()