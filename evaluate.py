import torch
import torch.nn.functional as F
from PIL import Image
import wandb
import torch.nn as nn
from unet import UNet, TemporalUNet, ConvLSTM, ConvGRU, TemporalTransformerUNet
from utils.data_loading import DSADataset
from torch.utils.data import DataLoader, random_split
from utils.metrics import dice_coeff, dice_loss, acc_sens_spec
import os
import numpy as np
from pathlib import Path
import pandas as pd
import time
import argparse
import logging
import sys

criterion = nn.CrossEntropyLoss()


def evaluate(net, dataloader, device=torch.device('cuda'), mode='unknown', wb=None, save=False):
    """
    Important Note: make sure the batch_size is set to one in the dataloader.
    Otherwise, the logging might be missing and the calculated values might be incorrect.
    """

    if len(dataloader) == 0:
        raise ValueError("Empty validation set.")

    net.eval()
    dices = []
    artery_dices = []
    vein_dices = []
    accuracies = []
    specificities = []
    sensitivities = []
    epoch_loss = 0
    results = []
    inf_times = []

    # iterate over the validation set
    for batch in dataloader:
        images, masks_true, filename = batch['image'], batch['mask'], batch['filename'][0]
        result = {'filename': filename}
        # move images and labels to correct device and type
        images = images.to(device=device, dtype=torch.float32)
        masks_true = masks_true.to(device=device, dtype=torch.long)

        # converting the true_masks from 4 classes to 2 classes
        if net.n_classes == 2:
            artery_masks = torch.logical_or(masks_true == 1, masks_true == 3)
            vein_masks = torch.logical_or(masks_true == 2, masks_true == 3)
            bg_masks = (masks_true == 0)
            masks_true = torch.stack((bg_masks, artery_masks, vein_masks), dim=1).float()
        if net.n_classes == 1:
            masks_true = F.one_hot(masks_true, 2).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            start = time.time()
            masks_pred = torch.sigmoid(net(images))
            end = time.time()
            inf_times.append(end - start)
            loss = criterion(masks_pred, masks_true[:, 1:, ...]) + dice_loss(masks_pred, masks_true[:, 1:, ...],
                                                                              multiclass=True)
            epoch_loss += loss.item()

            # convert to one-hot format
            if net.n_classes == 1:
                masks_pred_one_hot = torch.squeeze(masks_pred, 1) >= 0.5
                masks_pred_one_hot = F.one_hot(masks_pred_one_hot.to(torch.int64), 2).permute(0, 3, 1, 2).float()
                # compute the Dice score
                dice = dice_coeff(masks_pred_one_hot[:, 1, ...], masks_true[:, 1, ...], reduce_batch_first=False, multiclass=True)
                dices.append(dice)
                if save:
                    mask_vis = Image.fromarray(masks_pred_one_hot[0, 1, ...].cpu().detach().numpy().astype(np.uint8))
                    result.update({'dice': dice.item()})
            else:
                masks_pred_one_hot = (masks_pred >= 0.5)
                bg_masks_pred_one_hot = torch.logical_not(
                    torch.logical_or(masks_pred_one_hot[:, 0, ...], masks_pred_one_hot[:, 1, ...]))
                masks_pred_one_hot = torch.stack(
                    (bg_masks_pred_one_hot, masks_pred_one_hot[:, 0, ...], masks_pred_one_hot[:, 1, ...]),
                    dim=1).float()
                # compute the Dice score, ignoring background
                dice = dice_coeff(masks_pred_one_hot[:, 1:, ...], masks_true[:, 1:, ...], reduce_batch_first=False,
                                  multiclass=True)
                dices.append(dice)
                adice = dice_coeff(masks_pred_one_hot[:, 1, ...], masks_true[:, 1, ...], reduce_batch_first=False)
                artery_dices.append(adice)
                vdice = dice_coeff(masks_pred_one_hot[:, 2, ...], masks_true[:, 2, ...], reduce_batch_first=False)
                vein_dices.append(vdice)
                if save:
                    masks_av = torch.logical_or(masks_pred_one_hot[:, 1, ...], masks_pred_one_hot[:, 2, ...])
                    masks_artery = torch.logical_and(masks_pred_one_hot[:, 1, ...],
                                                     torch.logical_not(masks_pred_one_hot[:, 2, ...]))
                    masks_vein = torch.logical_and(torch.logical_not(masks_pred_one_hot[:, 1, ...]),
                                                   masks_pred_one_hot[:, 2, ...])
                    mask_vis = torch.stack((bg_masks_pred_one_hot, masks_artery, masks_vein, masks_av), dim=1)
                    mask_vis = Image.fromarray(mask_vis[0].permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8))
                    result.update({'dice': dice.item(), 'adice': adice.item(), 'vdice': vdice.item()})

            # compute the accuracy, including background
            acc = acc_sens_spec(masks_pred_one_hot, masks_true, reduce_batch_first=False, multiclass=True)
            accuracies.append(acc)
            # compute the sensitivity, excluding background
            sens = acc_sens_spec(masks_pred_one_hot[:, 1:, ...], masks_true[:, 1:, ...], reduce_batch_first=False,
                                 multiclass=True)
            sensitivities.append(sens)
            # compute the specificity, using only background
            spec = acc_sens_spec(masks_pred_one_hot[:, 0, ...], masks_true[:, 0, ...], reduce_batch_first=False,
                                 multiclass=True)
            specificities.append(spec)
            if save:
                result.update({'acc': acc.item(), 'sens': sens.item(), 'spec': spec.item()})
                results.append(result)
                save_path = os.path.join("./results", wb.name, '{}.png'.format(filename))
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                mask_vis.save(save_path)

            # if wb is not None:
            #     wb.log({
            #         '{}/true_{}'.format(mode, filename): wandb.Image(masks_true[0].float().cpu()),
            #         '{}/pred_{}'.format(mode, filename):
            #             wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu())})
            #     if images[0].dim() == 3:  # C*H*W
            #         wb.log({'{}/minip_{}'.format(mode, filename): wandb.Image(images[0].cpu())})
            #     else:  # T*C*H*W
            #         wb.log({
            #             '{}/sequence_{}'.format(mode, filename): wandb.Image(images[0].cpu()),
            #             '{}/minip_{}'.format(mode, filename): wandb.Image(torch.min(images[0], dim=0).values.cpu())})
    net.train()

    if save:
        df_result = pd.DataFrame.from_records(results)
        Path("./results").mkdir(parents=True, exist_ok=True)
        df_result.to_csv(os.path.join("./results", wb.name, 'results.csv'),
                         index=False, header=True, float_format='%.3f')

    dices = torch.stack(dices)
    dice_mean, dice_std = torch.mean(dices).item(), torch.std(dices).item()
    on_target_rate = torch.sum(dices > 0.05).item() / dices.numel()
    accuracies = torch.stack(accuracies)
    acc_mean, acc_std = torch.mean(accuracies).item(), torch.std(accuracies).item()
    specificities = torch.stack(specificities)
    spec_mean, spec_std = torch.mean(specificities).item(), torch.std(specificities).item()
    sensitivities = torch.stack(sensitivities)
    sens_mean, sens_std = torch.mean(sensitivities).item(), torch.std(sensitivities).item()

    summary_dict = {f'{mode}_on_target_rate': on_target_rate,
                    f'{mode}_dice_mean': dice_mean, '{}_dice_std'.format(mode): dice_std,
                    f'{mode}_acc_mean': acc_mean, '{}_acc_std'.format(mode): acc_std,
                    f'{mode}_spec_mean': spec_mean, '{}_spec_std'.format(mode): spec_std,
                    f'{mode}_sens_mean': sens_mean, '{}_sens_std'.format(mode): sens_std,
                    f'{mode}_inf_time_mean': np.mean(np.array(inf_times)),
                    f'{mode}_inf_time_std': np.std(np.array(inf_times))}
    if net.n_classes == 4:
        artery_dices, vein_dices = torch.stack(artery_dices), torch.stack(vein_dices)
        summary_dict['{}_adice_mean'.format(mode)] = torch.mean(artery_dices).item()
        summary_dict['{}_adice_std'.format(mode)] = torch.std(artery_dices).item()
        summary_dict['{}_vdice_mean'.format(mode)] = torch.mean(vein_dices).item()
        summary_dict['{}_vdice_std'.format(mode)] = torch.std(vein_dices).item()
    if wb is not None:
        wb.log(summary_dict)
    return dice_mean, epoch_loss / len(dataloader), summary_dict

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--input-type', '-i', default='minip', help='Model input - minip or sequence.')
    parser.add_argument('--label-type', '-t', default='vessel', help='Label type - vessel (binary) or av (2 classes).')
    parser.add_argument('--rnn', '-r', type=str, default='ConvGRU', help='RNN type: convGRU, convLSTM, or TemporalTransformer.')
    parser.add_argument('--rnn_kernel', '-k', type=int, default=1, help='RNN kernel: 1 (1x1) or 3 (3x3).')
    parser.add_argument('--rnn_layers', '-n', type=int, default=2, help='Number of RNN layers.')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of transformer attention heads.')
    parser.add_argument('--img_scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision.')
    parser.add_argument('--exp_group', '-g', type=str, default=None, help='Set wandb group name.')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S',
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    '''Global settings'''
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args()
    assert args.input_type == 'minip', "Invalid input image type"
    assert args.label_type == 'vessel', "Invalid label type"
    test_img_dir = os.path.join('./data/ICATopSeg', 'test', 'minip')
    test_mask_dir = os.path.join('./data/ICATopSeg', 'test', 'masks')
    exp_name = 'vocal-thunder-9'
    n_classes = (1, 2)[args.label_type == 'av']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    '''Set up the network'''
    if args.input_type == 'minip':
        net = UNet(n_channels=1, n_classes=n_classes, bilinear=True)
    else:
        if args.rnn in ['ConvLSTM', 'ConvGRU']:
            rnn = (ConvGRU, ConvLSTM)[args.rnn == 'ConvLSTM']
            kernel_size = (args.rnn_kernel, args.rnn_kernel)
            net = TemporalUNet(rnn, n_channels=1, kernel_size=kernel_size, num_layers=args.rnn_layers,
                               n_classes=n_classes, bilinear=True)
        elif args.rnn == "TemporalTransformer":
            net = TemporalTransformerUNet(n_channels=1, n_classes=n_classes,
                                        H=int(1024*args.img_scale), W=int(1024*args.img_scale), bilinear=True,
                                          num_layers=args.rnn_layers, num_heads=args.num_heads)
        else:
            raise ValueError("Unsupported rnn type!")

    net.to(device=device)

    '''1. Create dataset'''
    test_set = DSADataset(test_img_dir, test_mask_dir, scale=args.img_scale)

    '''2. Create data loaders'''
    loader_args = dict(num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, batch_size=1, **loader_args)

    '''3. Load a model'''
    # make sure wandb is set. It generates a random experiment name, which is used as the result folder name.
    ckpt = os.path.join("./results", exp_name, 'checkpoint.pt')
    Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
    net.load_state_dict(torch.load(ckpt))
    # torch.save(net, "./models/best_model_ica_top.pt")


    '''4. Begin testing'''
    experiment = wandb.init(project='ICATopSeg', name=exp_name, resume='allow', anonymous='must', group=args.exp_group)
    _, _, test_result = evaluate(net, test_loader, mode='test', device=device, wb=experiment, save=True)
    logging.info("Results (test) ---- {}".format(test_result))

    '''7. Write results to CSV files'''
    result = {'wandb': exp_name}
    result.update(vars(args))
    result.update(test_result)
    df_result = pd.DataFrame.from_records([result])
    Path("./results").mkdir(parents=True, exist_ok=True)
    df_result.to_csv(os.path.join("./results", exp_name, 'summary.csv'),
                     index=False, header=True, float_format='%.3f')
    logging.info("Done!")