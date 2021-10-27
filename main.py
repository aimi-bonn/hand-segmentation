from __future__ import print_function
import argparse
import datetime
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from lib.datasets.bonemask_dataset import BoneMaskDataset
from lib.models.EffUNet import EffUNet
from lib.models.UNet import UNet
from lib.utils import dice_score, BinaryDiceLoss, jaccard_index, TverskyLoss

saved_model_dir = "saved_models"
dataset_train_path_bonemask = '../data/BoneMask/train/'
dataset_val_path_bonemask = '../data/BoneMask/val/'


# Function that helps set each worker's seed differently (consistently)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=((1e-3)), metavar='LR', 
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-all', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--session', type=int, default=1, dest='session',
                        help='Session used to distinguish model tests.')
    parser.add_argument('--model-type', default='UNet', dest='model_type',
                        help='Model type to use. (Options: UNet, EffUNet)')
    parser.add_argument('--up-type', default='upsample', dest='up_type',
                        help='Upsampling type to use in UpConv part of UNet (Options: upsample, upconv)')
    parser.add_argument('--act-type', default='PReLU', dest='act_type',
                        help='activation function to use in UNet. (Options: ReLU, PReLU)')
    parser.add_argument('--norm', default='GN', dest='norm',
                        help='Which Normalization to use: None, BN, GN')

    # Which dataset do we use?
    parser.add_argument('--dataset', default='bonemask', dest='dataset',
                        help='Which dataset to use: bonemask')

    parser.add_argument('--use_tensorboard', action='store_true', default=False,
                        help='Use tensorboard for logging')

    parser.add_argument('--val-interval', type=int, default=500,
                        help='how many batches to wait before validation is evaluated (and optimizer is stepped).')

    return parser.parse_args()


def train(args, model, device, train_loader, optimizer, val_loader=None, scheduler=None, epochs=-1):
    model.train()

    # Time measurements
    tick = datetime.datetime.now()

    # Tensorboard Writer
    if args.use_tensorboard:
        writer = SummaryWriter(
            comment=f"s{args.session}_{args.model_type}_{args.up_type}_{args.act_type}_bs{args.batch_size}"
                    f"_{args.norm}")
    global_step = 0

    # Saving variables
    prev_dice = 0
    avg_val_dice = 0

    # These values were selected because of the ratio in which I expect fore- and background pixels during segmentation
    tversky_Loss = TverskyLoss(alpha=0.75, beta=0.25, apply_nonlin=torch.sigmoid)
    mse_loss = 0.
    if epochs == -1:
        epochs = args.epochs
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.
        for batch_idx, (data, targets) in enumerate(train_loader):

            data, target_mask = data.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32)
            output = model(data)

            bce_loss = F.binary_cross_entropy_with_logits(output, target_mask)
            t_loss = tversky_Loss(output, target_mask)
            mask_loss = bce_loss - torch.log(dice_score(torch.sigmoid(output), target_mask)) + t_loss
            loss = mask_loss
            
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            if (batch_idx + 1) % args.log_interval == 0:
                tock = datetime.datetime.now()
                print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t(Elapsed time {:.1f}s)'.format(
                    tock.strftime("%H:%M:%S"), epoch, batch_idx * len(data), len(train_loader.dataset),
                                                      100. * batch_idx / len(train_loader), loss.item(),
                    (tock - tick).total_seconds()))
                tick = tock

                if args.use_tensorboard:
                    writer.add_scalar('train/Mask_Loss', mask_loss.item(), global_step)

            if val_loader:
                if (batch_idx + 1) % args.val_interval == 0:
                    #avg_val_dice, avg_val_loss, avg_mse_loss = validate(model, device, val_loader)
                    avg_val_dice, avg_val_loss = validate(model, device, val_loader)

                    ## Only use when using a scheduler that needs output from validate; e.g. reduce on plateau
                    #if scheduler:
                    #    scheduler.step()

                    tick = datetime.datetime.now()

                    if args.use_tensorboard:
                        writer.add_scalar('val/Dice', avg_val_dice, global_step)
                        writer.add_scalar('val/Loss', avg_val_loss, global_step)

                        writer.add_images('images', data, global_step)
                        writer.add_images('masks/true', target_mask, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(output) > 0.5, global_step)

            global_step += args.batch_size

        ## Epoch is completed
        print(f"Overall average training loss: {epoch_loss / len(train_loader):.6f}")

        # Save model
        if args.save_all or avg_val_dice > prev_dice:
            print(
                f"Saving model in: {saved_model_dir}/s{args.session}_{args.dataset}_adam_augment_{args.model_type}_e{epoch}_{args.act_type}"
                f"_{args.up_type}_{args.norm}_bs{args.batch_size}.pt")
            torch.save(model.state_dict(),
                       f"{saved_model_dir}/s{args.session}_{args.dataset}_adam_augment_{args.model_type}_e{epoch}_{args.act_type}"
                       f"_{args.up_type}_{args.norm}_bs{args.batch_size}.pt")
            prev_dice = avg_val_dice

        # See what would be the best threshold for the DiceScore on our validation set
        validate(model, device, val_loader, eval_all_thresholds=True)

        if scheduler:
            scheduler.step()

    if args.use_tensorboard:
        writer.flush()
        writer.close()


def validate(model, device, val_loader, eval_all_thresholds=False):
    model.eval()
    criterion = BinaryDiceLoss()
    val_bce_loss = 0.
    val_dice_loss = 0.
    val_jacc_loss = 0.

    jacc_index = 0.
    dice_scores = 0.

    if eval_all_thresholds:
        thresh_range = np.arange(0, 100)
        dice_scores = np.zeros(len(thresh_range))

    tick = datetime.datetime.now()
    with torch.no_grad():
        for (data, targets) in val_loader:
            data, target_mask = data.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32)
            output = model(data)
            
            # Evaluate the dice score on all thresholds [0.00, 0.01, ..., 0.99]
            if eval_all_thresholds:
                for idx, thresh in enumerate(thresh_range):
                    output_temp = (torch.sigmoid(output) > (thresh / 100)).float()
                    dice_scores[idx] += dice_score(output_temp, target_mask).item()
            else:
                output = (torch.sigmoid(output) > 0.5).float()

                val_dice_loss += criterion(output, target_mask).item()

                jacc_index += jaccard_index(output, target_mask).item()
                val_jacc_loss += (1 - jaccard_index(output, target_mask).item())

                dice_scores += dice_score(output, target_mask).item()

    val_loss = val_dice_loss + val_bce_loss
    model.train()

    if eval_all_thresholds:
        for idx, thresh in enumerate(thresh_range):
            print(f"\tThreshold {thresh / 100} (index {idx}) "
                  f"resulted in a DiceScore of {dice_scores[idx] / len(val_loader)}")
    else:
        print(f"Average BCE ({val_bce_loss / len(val_loader)}), "
              f"JaccardLoss ({val_jacc_loss / len(val_loader)}), "
              f"DiceLoss ({val_dice_loss / len(val_loader)}) and "
              f"Overall Loss ({val_loss / len(val_loader)}) during validation")
        print(f"Average Dice score during validation: {dice_scores / len(val_loader)}, "
              f"average Jaccard index: {jacc_index / len(val_loader)}")
        print(f"Elapsed time during validation: {(datetime.datetime.now() - tick).total_seconds():.1f}s")

    return dice_scores / len(val_loader), val_bce_loss / len(val_loader)


def main():
    # Training settings
    args = parse_args()
    
    # Create the save directory for model weights
    os.makedirs(saved_model_dir, exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)

    kwargs = {}
    if use_cuda:
        kwargs.update({'pin_memory': True})

    if args.dataset == 'bonemask':
        in_channels = 1
        dataset_train = BoneMaskDataset(imgs_dir=dataset_train_path_bonemask,
                                        img_suffix='_ori',
                                        mask_suffix='_bin_pruned',
                                        size=512)
        dataset_val = BoneMaskDataset(imgs_dir=dataset_val_path_bonemask,
                                      img_suffix='_ori',
                                      mask_suffix='_bin_pruned',
                                      size=512,
                                      validation=True)
        
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                   worker_init_fn=seed_worker, **kwargs)
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.test_batch_size, shuffle=False,
                                                 num_workers=0, worker_init_fn=seed_worker, **kwargs)
    else:
        print("No valid dataset requested")
        raise NotImplementedError()

    if args.norm == 'BN':
        norm_type = nn.BatchNorm2d
    elif args.norm == 'GN':
        norm_type = nn.GroupNorm
    elif args.norm == 'None':
        norm_type = None
    else:
        print(f"Unknown Normalization type given: {args.norm}")
        raise NotImplementedError()
    print(f"Using Normalization type: {norm_type}")

    if args.act_type == "ReLU":
        act_type = nn.ReLU
    elif args.act_type == "PReLU":
        act_type = nn.PReLU
    else:
        print(f"Invalid ACT_type given! (Got {args.ACT_type})")
        raise NotImplementedError()

    if args.model_type == 'UNet':
        model = UNet(depth=5, in_channels=in_channels, num_classes=1, padding=1, act_type=act_type, norm_type=norm_type,
                     up_type=args.up_type).to(device)
    elif args.model_type == 'EffUNet':
        eff_version = "efficientnet-b0"
        model = EffUNet(in_channels=in_channels, act_type=act_type, norm_type=norm_type, #nn.BatchNorm2d,
                        up_type=args.up_type, base=eff_version).to(device)
    else:
        print(f"No valid model type given! (got model_type: {args.model_type})")
        raise NotImplementedError()


    ## Number of images before running validation:
    args.log_interval = args.log_interval // args.batch_size
    args.val_interval = 237 // args.batch_size  # not so sure why I'm hardcoding that here .. 

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr*10, momentum=0.99, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train(args, model, device, train_loader, optimizer, val_loader=val_loader, scheduler=scheduler)


if __name__ == '__main__':
    main()
