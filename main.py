import numpy as np
import torch
import torch.nn as nn
import os
import os.path as osp
import configargparse

import trainer
import data_loader
from model import DenseNet103

def mkdir_f(d):
    if not os.path.exists(f):
        os.mkdir(f)

def symlink_train_test(input_image_dir: str,
                  image_glob: str,
                  input_mask_dir: str,
                  mask_glob: str,) -> None:
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
    imgs = sorted(glob.glob(osp.join(input_image_dir, image_glob)))
    masks = sorted(glob.glob(osp.join(input_mask_dir, mask_glob)))
    assert len(imgs) == len(masks), \
        'imgs %d and masks %d have unequal numbers' % (len(imgs), len(masks))
    print('Loaded imgs %d and masks %d.'% (len(imgs), len(masks)))

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
    mkdir_f(train_path)
    mkdir_f(test_path)
    for tp in [train_path, test_path]:
        for d in ['images', 'masks']:
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
    return

def train(args):
    assert args.output_path is not None
    assert args.input_image_dir is not None
    assert args.input_mask_dir is not None

    if args.loss.lower() == 'dice':
        criterion = trainer.dice_loss_integer
    elif args.loss.lower() == 'focal':
        criterion = trainer.FocalLoss()

    if args.transform.lower() == 'crop512raw':
        transform = data_loader.crop512raw
    elif args.transform.lower() == 'crop512':
        transform = data_loader.crop512

    training_set_fraction = 0.90

    symlink_train_test(args.input_image_dir,
                       args.image_glob,
                       args.input_mask_dir,
                       args.mask_glob)

    train_ds = data_loader.CellDataset(osp.join(train_path, 'images'),
                           osp.join(train_path, 'masks'),
                           transform=transform,
                           symlinks=True)

    test_ds = data_loader.CellDataset(osp.join(test_path, 'images'),
                          osp.join(test_path, 'masks'),
                          transform=transform,
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
    model = DenseNet103(n_classes=args.n_classes)
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        print('No CUDA available, running on CPU!')
    print('Model loaded.')

    optimizer = optim.RMSprop(model.parameters(),
                        lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                    step_size=1, gamma=0.995)

    print('Training')
    trainer = trainer.Trainer(model,
                      criterion,
                      optimizer,
                      dataloaders,
                      args.output_path,
                      n_epochs=300,
                      ignore_index = 3,
                      scheduler=scheduler,
                      verbose=False,
                      viz=False)
    trainer.train()

def predict(args):
    raise NotImplementedError()

def main():
    parser = configargparse.ArgParser('Utilize FC-DenseNet PyTorch models.',
        default_config_file=['./default_config.txt'])
    parser.add_argument('--config', is_config_file=True)
    parser.add_argument('--command', required=True, type=str,
        help='action to perform. {train, predict}')
    parser.add_argument('--input_image_dir', type=str, default=None)
    parser.add_argument('--image_glob', type=str, default='*.tif')
    parser.add_argument('--input_mask_dir', type=str, default=None)
    parser.add_argument('--mask_glob', type=str, default='*.png')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--loss', type=str, default='dice')
    parser.add_argument('--transform', type=str, default='crop512raw')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_weights', type=str, default=None,
        help='path to trained model weights. Required for prediction.')
    args = parser.parse_args()

    if args.command.lower() == 'train':
        train_model(args)
    elif args.command.lower() == 'predict':
        predict(args)
    return

if __name__ == '__main__':
    main()
