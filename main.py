import numpy as np
import torch
import torch.nn as nn
import os
import os.path as osp
import glob
import configargparse
import datetime

def mkdir_f(f):
    if not os.path.exists(f):
        os.mkdir(f)
        
from scipy.misc import imresize
from skimage.morphology import remove_small_objects, binary_erosion, binary_dilation, disk
from skimage.io import imsave

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

def symlink_train_test(input_image_dir: str,
                  image_glob: str,
                  input_mask_dir: str,
                  mask_glob: str,
                  expname: str) -> None:
    '''Copy symlinks to `train` and `test` subdirs within the
    input directories. This allows for random training splits without
    copying files.

    Parameters
    ----------
    input_image_dir : str
        path to input image files.
    input_mask_dir : str
        path to input mask files.
    '''
    print('Searching for images in\n %s' % input_image_dir)
    print('Searching for masks in\n %s' % input_mask_dir)

    training_set_fraction = 0.90
    imgs = sorted(glob.glob(osp.join(input_image_dir, image_glob)))
    masks = sorted(glob.glob(osp.join(input_mask_dir, mask_glob)))
    assert len(imgs) == len(masks), \
        'imgs %d and masks %d have unequal numbers' % (len(imgs), len(masks))
    print('Loaded imgs %d and masks %d.'% (len(imgs), len(masks)))
    assert len(imgs) > 0, 'no images/masks found!'

    data_dir = osp.split(input_image_dir)[0] # image dir parent will be used for traintest
    
    # Choose random indices for the train set
    n_train = int(np.floor(training_set_fraction*len(imgs)))
    train_idx = np.random.choice(
        np.arange(len(imgs)),
        size=n_train,
        replace = False).astype(np.int)
    test_idx = np.setdiff1d(np.arange(len(imgs)),
                            train_idx).astype(np.int)

    train_path = osp.join(data_dir, 'train_'+expname)
    test_path = osp.join(data_dir, 'test_'+expname)
    
    img_dir_prefix = 'images'
    mask_dir_prefix = 'masks'
    
    mkdir_f(train_path)
    mkdir_f(test_path)
    for tp in [train_path, test_path]:
        for d in [img_dir_prefix, mask_dir_prefix]:
            mkdir_f(osp.join(tp, d))

    print('Clearing train test...')
    old_symlinks = glob.glob(osp.join(train_path, '*', '*')) + glob.glob(osp.join(test_path, '*', '*'))
    for f in old_symlinks:
        if os.path.islink(f):
            os.remove(f)
    for i in range(len(train_idx)):
        os.symlink(imgs[train_idx[i]],
            osp.join(train_path, img_dir_prefix, osp.split(imgs[train_idx[i]])[-1]))
        os.symlink(masks[train_idx[i]],
            osp.join(train_path, mask_dir_prefix, osp.split(masks[train_idx[i]])[-1]))

    for i in range(len(test_idx)):
        os.symlink(imgs[test_idx[i]],
            osp.join(test_path, img_dir_prefix, osp.split(imgs[test_idx[i]])[-1] ))
        os.symlink(masks[test_idx[i]],
            osp.join(test_path, mask_dir_prefix, osp.split(masks[test_idx[i]])[-1] ))
    print('Symlinking finished.')
    return train_path, test_path

def train_model(args):
    import trainer
    import data_loader
    from model import DenseNet103

    assert args.output_path is not None
    assert args.input_image_dir is not None
    assert args.input_mask_dir is not None

    if args.loss.lower() == 'dice':
        criterion = trainer.DiceLoss(C=args.n_classes)
        val_occupied_only = True
    elif args.loss.lower() == 'focal':
        criterion = trainer.FocalLoss(size_average=True)
        val_occupied_only = False

    if args.transform.lower() == 'crop512raw':
        transform_train = data_loader.crop512raw
        transform_test_pre   = None
        transform_test_post  = data_loader.predcrop512
    elif args.transform.lower() == 'crop512':
        transform_train = data_loader.crop512
        transform_test_pre   = data_loader.predcrop512_pre
        transform_test_post  = data_loader.predcrop512

    if args.exp_name is not None:
        exp_name = args.exp_name
    else:
        exp_name = datetime.datetime.today().strftime('%Y%m%d')
        
    train_path, test_path = symlink_train_test(
                       args.input_image_dir,
                       args.image_glob,
                       args.input_mask_dir,
                       args.mask_glob,
                       exp_name)

    train_ds = data_loader.CellDataset(osp.join(train_path, 'images'),
                           osp.join(train_path, 'masks'),
                           transform=transform_train,
                           symlinks=True)

    # use a consistent set of crops in the testing set
    test_ds = data_loader.CellCropDataset(osp.join(test_path, 'images'),
                          osp.join(test_path, 'masks'),
                          transform_pre=transform_test_pre,
                          transform_post=transform_test_post,
                          n_windows=16,
                          symlinks=True)

    train_dl = torch.utils.data.DataLoader(train_ds,
                    batch_size=args.batch_size,
                    num_workers=6,
                    shuffle=True)
    test_dl  = torch.utils.data.DataLoader(test_ds,
                    batch_size=args.batch_size,
                    num_workers=6,
                    shuffle=False)
    dataloaders = {'train': train_dl,
                    'val':test_dl}

    print('Loading model...')
    model = DenseNet103(n_classes=args.n_classes,
                        growth_rate=args.growth_rate)
    if torch.cuda.is_available():
        model = model.cuda()
        print('Model moved to CUDA compute device.')
    else:
        print('No CUDA available, running on CPU!')
    print('Model loaded.')

    optimizer = torch.optim.RMSprop(model.parameters(),
                        lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                    step_size=1, gamma=0.995)
    
    mkdir_f(args.output_path)
    mkdir_f(osp.join(args.output_path, exp_name))
    exp_out_path = osp.join(args.output_path, exp_name)
    

    print('Training')
    trainer = trainer.Trainer(model,
                      criterion,
                      optimizer,
                      dataloaders,
                      exp_out_path,
                      n_epochs=args.n_epochs,
                      ignore_index = args.n_classes+1,
                      scheduler=scheduler,
                      verbose=True,
                      viz=True,
                      val_occupied_only=val_occupied_only)
    trainer.train()

def predict(args):
    import data_loader
    from model import DenseNet103, Ensemble
    
    # Get image names
    img_files = sorted(glob.glob(
        os.path.join(args.input_image_dir, args.image_glob)))
    # basenames w/o extensions
    img_names = [osp.splitext(osp.basename(x))[0] for x in img_files]
    
    # Load models and build ensemble
    models = []
    for m in model_paths:
        mdl = DenseNet103(n_classes=args.n_classes)
        mdl.load_state_dict(torch.load(m))
        if torch.cuda.is_available():
            mdl = mdl.cuda()
            print('Moved model to CUDA compute device.')
        else:
            print('No CUDA device available. Using CPU.')
        mdl.train(False)
        models.append(mdl)

    ens = Ensemble(models)
    print('Models loaded.')
    
    if args.transform.lower() == 'crop512raw':
        transform_pre   = None
        transform_post  = data_loader.predcrop512
    elif args.transform.lower() == 'crop512':
        transform_pre   = data_loader.predcrop512_pre
        transform_post  = data_loader.predcrop512
    
    pl = PredCropLoader(args.input_image_dir, 
                        transform_pre=tranform_pre,
                        transform_post=transform_post, 
                        dtype='uint16', 
                        img_regex=args.img_glob,
                        n_windows=16)

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
                if torch.cuda.is_available():
                    input_panel = input_panel.cuda()
                input_panel.requires_grad = False
                output = ens(input_panel)
                probs = sm(output)
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
        imsave(os.path.join(args.output_path, img_names[i] + '_densenet.png'), maskR*255)
        if verbose:
            print('Processed ', img_names[i])
        end = time.time()
        times.append(end-start)

    print('Average image processing time : ', np.mean(times))

    return
    
def main():
    parser = configargparse.ArgParser('Utilize FC-DenseNet PyTorch models.',
        default_config_files=['./default_config.txt'])
    parser.add_argument('--config', is_config_file=True,
        help='path to a configuration file.')
    parser.add_argument('--command', required=True, type=str,
        help='action to perform. {train, predict}.')
    parser.add_argument('--exp_name', type=str, default=None,
        help='name of the experiment. defaults to the date in YYYYMMDD format.')
    parser.add_argument('--input_image_dir', type=str, default=None,
        help='path to input image files')
    parser.add_argument('--image_glob', type=str, default='*.tif',
        help='pattern to match image files.')
    parser.add_argument('--input_mask_dir', type=str, default=None,
        help='path to input mask files.')
    parser.add_argument('--mask_glob', type=str, default='*.png',
        help='pattern to match mask files.')
    parser.add_argument('--output_path', type=str, default=None,
        help='path for training or prediction outputs.')
    parser.add_argument('--loss', type=str, default='focal',
        help='loss function for model training. one of ["dice", "focal"].')
    parser.add_argument('--growth_rate', type=int, default=16,
        help='number of feature maps to add in each DenseBlock.')
    parser.add_argument('--n_classes', type=int, default=2,
        help='number of classes in the target masks.')
    parser.add_argument('--transform', type=str, default='crop512',
        help='tranformations to apply to input data. one of ["crop512", "crop512raw"].')
    parser.add_argument('--batch_size', type=int, default=1,
        help='bathc size for training.')
    parser.add_argument('--n_epochs', type=int, default=500,
        help='number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4,
        help='learning rate for training using the RMSprop optimizer.')
    parser.add_argument('--model_weights', type=str, default=None,
        help='path to trained model weights. Required for prediction.')
    parser.add_argument('--upsamp_sz', type=int, nargs=2, default=[2110, 2492], 
        help='final size of upsampled mask.')
    args = parser.parse_args()

    if args.command.lower() == 'train':
        train_model(args)
    elif args.command.lower() == 'predict':
        predict(args)
    return

if __name__ == '__main__':
    main()
