'''
Train a PyTorch model
'''
import os
import shutil
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import model as mdls
import torchvision.utils as vutils
import utils
from skimage.io import imsave

# Inspired by W. Kentaro (@wkentaro)
def crossentropy2d(pred, target, weight=None, ignore_index=2, size_average=True):
    '''
    Parameters
    -----------
    pred : autograd.Variable. (N, C, H, W)
        where C is number of classes.
    target : (N, H, W), where all values 0 <= target[i] <= C-1.

    Returns
    -------
    loss : Tensor.
    '''
    # pred dims (N, C, H, W)
    n, c, h, w = pred.size()
    # log_p : log_probabilities (N, C, H, W)
    log_p = F.log_softmax(pred)
    # Linearize log_p
    # log_p : (N, C, H, W) --> (N*H*W, C)
    # move dim C over twice (N, C, H, W) > (N, H, C, W) > (N, H, W, C)
    log_p = log_p.transpose(1,2).transpose(2,3).contiguous()
    # (N, H, W, C) --> (N*H*W, C)
    log_p = log_p.view(-1, c)

    # Reshape target to a (N*H*W,), where each values 0 <= i <= C-1
    target = target.view(-1)
    loss = F.nll_loss(log_p, target, weight=weight,
                        size_average=True, ignore_index=ignore_index)

    return loss

# Dice loss from Roger Trullo
#

class DiceLoss(nn.Module):

    def __init__(self, ignore_label: int=3, C: int=3) -> None:
        '''
        Computes a Dice loss from 2D input of class scores and a target of integer labels.

        Parameters
        ----------
        ignore_label : integer.
            Must be final label in the sequence (TODO, generalize).
        C : integer.
            number of classes (including an ignored label if present!)

        Notes
        -----
        Credit to Roger Trullo
        https://github.com/rogertrullo/pytorch/blob/rogertrullo-dice_loss/torch/nn/functional.py#L708
        '''
        super(DiceLoss, self).__init__()
        self.ignore_label = ignore_label
        self.C = C
        return

    def forward(self, input_, target) -> float:
        target = utils.make_one_hot(target, C=self.C)
        # subindex target without the ignore label
        target = target[:,:self.ignore_label,...]

        assert input_.size() == target.size(), "Input sizes must be equal."
        assert input_.dim() == 4, "Input must be a 4D Tensor."

        probs=F.softmax(input_, dim=1)
        num=probs*target#b,c,h,w--p*g
        num=torch.sum(num,dim=3)#b,c,h
        num=torch.sum(num,dim=2)

        den1=probs*probs#--p^2
        den1=torch.sum(den1,dim=3)#b,c,h
        den1=torch.sum(den1,dim=2)

        den2=target*target#--g^2
        den2=torch.sum(den2,dim=3)#b,c,h
        den2=torch.sum(den2,dim=2)#b,c

        dice=2*(num/(den1+den2))
        dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

        dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

        return dice_total

def dice_loss_integer(input_, target, ignore_label=3, C=3):
    """
    Computes a Dice loss from 2D input of class scores and a target of integer labels.

    Parameters
    ----------
    input : torch.autograd.Variable
        size B x C x H x W representing class scores.
    target : torch.autograd.Variable
        integer label representation of the ground truth, same size as the input.

    Returns
    -------
    dice_total : float.
        total dice loss.
    """
    target = utils.make_one_hot(target, C=C)
    # subindex target without the ignore label
    target = target[:,:ignore_label,...]

    assert input_.size() == target.size(), "Input sizes must be equal."
    assert input_.dim() == 4, "Input must be a 4D Tensor."

    probs=F.softmax(input_, dim=1)
    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)


    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=3)#b,c,h
    den1=torch.sum(den1,dim=2)


    den2=target*target#--g^2
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c


    dice=2*(num/(den1+den2))
    dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

    dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

    return dice_total

from torch.autograd import Variable

class FocalLoss(nn.Module):
    '''
    Credit:
    https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    '''
    def __init__(self, gamma=2., alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,)): # long is removed in Python3
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1) # N,C,H,W => N,C,H*W
            input = input.transpose(1,2) # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2)) # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def tensor_norm(T):
    return (T - T.min())/(T-T.min()).max()

class Trainer(object):
    '''
    Trains a model
    '''

    def __init__(self,
                model,
                criterion,
                optimizer,
                dataloaders: dict,
                out_path: str,
                n_epochs: int=50,
                ignore_index: int=2,
                use_gpu: bool=torch.cuda.is_available(),
                verbose: bool=False,
                save_freq: int=10,
                scheduler = None,
                viz: bool=False,
                val_occupied_only: bool=False):

        '''
        Trains a PyTorch `nn.Module` object provided in `model`
        on training and testing sets provided in `dataloaders`
        using `criterion` and `optimizer`.

        Saves model weight snapshots every `save_freq` epochs and saves the
        weights with the best testing loss at the end of training.

        Parameters
        ----------
        model : torch model object, with callable `forward` method.
        criterion : callable taking inputs and targets, returning loss.
        optimizer : torch.optim optimizer.
        dataloaders : dict. train, val dataloaders keyed 'train', 'val'.
        out_path : string. output path for best model.
        n_epochs : integer. number of epochs for training.
        ignore_index : integer. class index to ignore, [0, n_class-1].
        use_gpu : boolean. use CUDA acceleration.
        verbose : boolean. write all batch losses to stdout.
        save_freq : integer. Number of epochs between model checkpoints. Default = 10.
        scheduler : learning rate scheduler.
        viz: bool. save visualizations of segmentation performance with print outputs.
        print_iter : int
            number of iterations between print outputs.
        val_occupied_only : bool
            if True, calculate validation statistics only for panels where
            foreground classes are actually present.
        '''
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.dataloaders = dataloaders
        self.out_path = out_path
        self.use_gpu = use_gpu
        self.ignore_index = ignore_index
        self.verbose = verbose
        self.save_freq = save_freq
        self.best_acc = 0.
        self.best_loss = 1.0e10
        self.scheduler = scheduler
        self.viz = viz
        self.print_iter = 50
        # only save validation metrics for subpanels that have foreground classes
        self.val_occupied_only = True

        if not os.path.exists(self.out_path):
            os.mkdir(self.out_path)
        # initialize log
        self.log_path = os.path.join(self.out_path, 'log.csv')
        with open(self.log_path, 'w') as f:
            header = 'Epoch,Iter,Running_Loss,Mode\n'
            f.write(header)

    def _save_train_viz(self,
                        inputs: torch.Tensor,
                        labels: torch.Tensor,
                        outputs: torch.Tensor,
                        iteration: int) -> None:
        '''save visualizations of training'''
        I = inputs.cpu().detach().numpy()
        L = labels.cpu().detach().numpy()
        O = outputs.cpu().detach().numpy()

        for b in range(1):
            imsave(os.path.join(self.out_path,
                    self.training_state + '_inputs_e' \
                                + str(self.epoch).zfill(4) \
                                + '_i' + str(iteration).zfill(4) + '.png'),
                  (np.squeeze(I[b,0,...])*255).astype(np.uint8))
            imsave(os.path.join(self.out_path,
                    self.training_state + '_labels_e' \
                                + str(self.epoch).zfill(4) \
                                + '_i' + str(iteration).zfill(4) + '.png'),
                  (np.squeeze(L[b,0,...])*255).astype(np.uint8))
            imsave(os.path.join(self.out_path,
                    self.training_state + '_outputs_e' \
                                + str(self.epoch).zfill(4) \
                                + '_i' + str(iteration).zfill(4) + '.png'),

                  (np.squeeze(O[b,...])*255).astype(np.uint8))

    def train_epoch(self) -> None:
        # Run a train and validation phase for each epoch
        self.model.train(True)
        i = 0
        running_loss = 0.0
        for data in self.dataloaders['train']:
            inputs, labels = data['image'], data['mask']
            if self.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                pass
            inputs.requires_grad = True
            labels.requires_grad = False


            # zero gradients
            self.optimizer.zero_grad()

            # forward pass
            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, 1)
            if self.verbose:
                print('output size: ', outputs.size())
                self.last_output = outputs
                print('preds size: ', preds.size())

            loss = self.criterion(outputs, labels)
            if self.verbose:
                print('batch loss: ', loss.item())
            assert np.isnan(loss.data.cpu().numpy()) == False, 'NaN loss encountered in training'

            # backward pass
            loss.backward()
            self.optimizer.step()

            # statistics update
            running_loss += loss.detach().item() / inputs.size(0)

            if i % self.print_iter == 0:
                print('Iter : ', i)
                print('running_loss : ', running_loss / (i + 1))
                # append to log
                with open(self.log_path, 'a') as f:
                    f.write(str(self.epoch) + ',' + str(i) + ',' + str(running_loss / (i + 1)) + ',train\n')
                if self.viz:
                    self._save_train_viz(inputs, labels, preds, i)
            i += 1

        epoch_loss = running_loss / len(self.dataloaders['train'])
        # append to log
        with open(self.log_path, 'a') as f:
            f.write(str(self.epoch) + ',' + str(i) + ',' + str(running_loss / (i + 1)) + ',train_epoch\n')

        print('{} Loss : {:.4f}'.format('train', epoch_loss))

    def val_epoch(self) -> None:
        self.model.eval()
        self.optimizer.zero_grad()
        i = 0
        running_loss = 0.0
        running_corrects = 0.0
        running_total = 0.0
        counted_panels = 0
        for data in self.dataloaders['val']:
            inputs, labels = data['image'], data['mask']
            if self.use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()
            else:
                pass
            inputs.requires_grad = False
            labels.requires_grad = False # just double check these are volatile

            # forward pass
            outputs = self.model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = self.criterion(outputs, labels)
            if self.verbose:
                print('batch loss: ', loss.item())

            # statistics update
            if labels.sum() > 0 or (not self.val_occupied_only):
                running_loss += loss.detach().item() / inputs.size(0)
                counted_panels += 1

            if i % self.print_iter == 0:
                print('Iter : ', i)
                print('running_loss : ', running_loss / (i + 1))
                # append to log
                with open(self.log_path, 'a') as f:
                    f.write(str(self.epoch) + ',' + str(i) + ',' + str(running_loss / (i + 1)) + ',val\n')
                if self.viz:
                    self._save_train_viz(inputs, labels, preds, i)
            i += 1

        epoch_loss = running_loss / counted_panels
        # append to log
        with open(self.log_path, 'a') as f:
            f.write(str(self.epoch) + ',' + str(i) + ',' + str(running_loss / (i + 1)) + ',val_epoch\n')

        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.best_model_wts = self.model.state_dict()
            torch.save(self.model.state_dict(), os.path.join(self.out_path, 'model_weights_' + str(self.epoch).zfill(3) + '.pickle'))
        elif (self.epoch%self.save_freq == 0):
            torch.save(self.model.state_dict(),
                       os.path.join(self.out_path, 'model_weights_' + str(self.epoch).zfill(3) + '.pickle'))

        print('{} Loss : {:.4f}'.format('val', epoch_loss))


    def train(self) -> nn.Module:
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            print('Epoch {}/{}'.format(epoch, self.n_epochs - 1))
            print('-' * 10)
            # run training epoch
            self.training_state = 'train'
            self.train_epoch()
            self.training_state = 'val'
            with torch.no_grad():
                self.val_epoch()
            if self.scheduler is not None:
                self.scheduler.step()                

        print('Saving best model weights...')
        torch.save(self.model.state_dict(), os.path.join(self.out_path, '00_best_model_weights.pickle'))
        print('Saved best weights.')
        self.model.load_state_dict(self.best_model_wts)
        return self.model
