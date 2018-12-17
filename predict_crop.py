'''Run predictions'''
import sys
sys.path = sys.path + ['/home/jacob/src/fcdensenet_pytorch/']
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from model import DenseNet103
from data_loader import PredCropLoader, predcrop512, predcrop512_pre
import os
import os.path as osp
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


def post_process(sm_out, min_sm_prob=0.50, sz_min=2000, cell_idx=1, edge_idx=None, erod=False, dil=False):
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
    if edge_idx is not None:
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
    parser = argparse.ArgumentParser(description='Segment images using trained FC-DenseNet models')
    parser.add_argument('in_dir', type=str, help='image directory of input images')
    parser.add_argument('out_dir', type=str, help='output directory for segmented images')
    parser.add_argument('models', type=str, nargs='+', help='paths to model weights. multiple models will be Ensembled.')
    parser.add_argument('--img_glob', type=str, default='*', help='pattern to glob image filenames in `in_dir`')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes in the output masks')
    parser.add_argument('--min_smp', type=float, default=0.5, help='minimum softmax score for object pixels')
    parser.add_argument('--sz_min', type=float, default=10., help='minimum object size for post-processing')
    parser.add_argument('--erod', default=False, help='size of disk selem for post-processing erosion')
    parser.add_argument('--dil', default=False, help='size of disk selem for post-processing dilation')
    parser.add_argument('--upsamp_sz', type=int, nargs=2, default=[2110, 2492], help='final size of upsampled mask')
    parser.add_argument('--quiet', action='store_true', default=False, help='suppress verbose output')

    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    model_paths = args.models
    img_glob = args.img_glob
    min_smp = args.min_smp
    sz_min = args.sz_min
    erod = int(args.erod)
    dil = int(args.dil)
    upsamp_sz = tuple(args.upsamp_sz)
    verbose = np.logical_not(args.quiet)

    # Get image names
    img_files = sorted(glob.glob(os.path.join(in_dir, img_glob)))
    # names w/o extensions
    img_names = [osp.splitext(osp.basename(x))[0] for x in img_files] 

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
    tr_pre = predcrop512_pre
    tr_post = predcrop512
    pl = PredCropLoader(in_dir, 
                        transform_pre=tr_pre,
                        transform_post=tr_post, 
                        dtype='uint16', 
                        img_regex=img_glob)
    print('Data loader initialized.')
    print(len(pl), ' images found.')
    # initialize softmax fnx
    sm = torch.nn.Softmax2d()

    # Segment images
    print('Segmenting...')
    times = []
    for i in range(len(pl)):
        start = time.time()
        
        samples = pl[i] 
        print('Number of samples %d' % len(samples))
        
        outs = [] # np.ndarrays of softmax probs
        with torch.no_grad():
            for d in samples:
                input_panel = d['image'].unsqueeze(0) # add empty batch
                print('Input panel size ', input_panel.size())
                o = predict(ens, input_panel)
                probs = sm(o)
                probs = probs.cpu().data.numpy() # unpack to numpy array
                outs.append(probs)
            
        # call a mask for each set of probs
        mask_panels = []
        for sm_panel in outs:
            mask = post_process(sm_panel, 
                                min_sm_prob=min_smp, 
                                sz_min=sz_min, 
                                erod=erod, 
                                dil=dil)
            mask_panels.append(mask)

        # reconstruct total mask from masks
        mask_sz = mask_panels[0].shape
        n_per_side = int(np.sqrt(len(mask_panels)))
        total_mask = np.zeros((mask_sz[0]*n_per_side, mask_sz[1]*n_per_side))
        for j in range(len(mask_panels)):
            total_mask[(j//n_per_side)*mask_sz[0]:((j//n_per_side)+1)*mask_sz[0],
                       (j% n_per_side)*mask_sz[1]:((j% n_per_side)+1)*mask_sz[1]] = mask_panels[j]
        
            
            
        maskR = imresize(total_mask.astype('uint8'), upsamp_sz, interp='nearest') # upsample
        maskR = maskR.astype('bool').astype('uint8')
        # save upsamples mask
        imsave(os.path.join(out_dir, img_names[i] + '_densenet.png'), maskR*255)
        if verbose:
            print('Processed ', img_names[i])
        end = time.time()
        times.append(end-start)

    print('Average image processing time : ', np.mean(times))


    return

if __name__ == '__main__':
    main()
