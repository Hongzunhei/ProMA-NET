"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019.

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu.
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import argparse
import time
import numpy as np
import torch
from ProMA_net import generators, layers, losses, utils
import ml_collections
import ProMA_net.ProMA_NET as ProMA_NET
import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)

def dice_coefficient(mask1, mask2):
    mask1 = np.round(mask1)
    mask2 = np.round(mask2)
    intersection = np.logical_and(mask1, mask2).sum()
    volume_sum = mask1.sum() + mask2.sum()
    if volume_sum == 0:
        return 1.0
    dice = 2.0 * intersection / volume_sum
    return dice


def get_3DTransMorph_config():
    '''
    Trainable params: 15,201,579
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 1
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (4, 4, 4)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.rpe = True
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (128, 128, 128)
    return config

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', default='./train_model/models_PMA',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

parser.add_argument('--gpu', default='1', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=60,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--networks', default='unet', help='choose which network to train,unet or localnet')
parser.add_argument('--load-model',default=None, help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.3,
                    help='weight of deformation loss (default: 0.3)')
parser.add_argument('--dice-weight', type=float, default=1,
                    help='weight of the DICE loss (default: 1.0)')


parser.add_argument('--data-dir', default='/home/data1/kjx/sci/voxelmorph-dev/data/', help='root directory of the dataset')
args = parser.parse_args()
bidir = args.bidir

add_feat_axis = not args.multichannel
train_fixed_images_dir, train_moving_images_dir, train_fixed_labels_dir, train_moving_labels_dir, \
val_fixed_images_dir, val_moving_images_dir, val_fixed_labels_dir, val_moving_labels_dir, _, _, _, _ = generators.get_dataset_paths(args.data_dir)

generator = generators.paired_scan_generator(
    fixed_image_dir=train_fixed_images_dir,
    moving_image_dir=train_moving_images_dir,
    fixed_label_dir=train_fixed_labels_dir,
    moving_label_dir=train_moving_labels_dir,
    batch_size=args.batch_size,
    add_feat_axis=True
)

# Validation data generator
val_generator = generators.paired_scan_generator(
    fixed_image_dir=val_fixed_images_dir,
    moving_image_dir=val_moving_images_dir,
    fixed_label_dir=val_fixed_labels_dir,
    moving_label_dir=val_moving_labels_dir,
    batch_size=args.batch_size,
    add_feat_axis=True
)
sample_input = next(generator)
inshape = sample_input[0][0].shape[1:-1]
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)
gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size, nb_gpus) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

if args.load_model:
    config = get_3DTransMorph_config()
    model = ProMA_NET.ProMANET.load(args.load_model, device)
else:
    config = get_3DTransMorph_config()
    model = ProMA_NET.ProMANET(config)

if nb_gpus > 1:
    model = torch.nn.DataParallel(model)
    model.save = model.module.save


model.to(device)
model.train()
torch.autograd.set_detect_anomaly(True)
stn = layers.SpatialTransformer(inshape).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.image_loss == 'ncc':
    image_loss_func = losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = losses.MSE().loss
elif args.image_loss == 'mind':
    image_loss_func = losses.MIND().loss
elif args.image_loss == 'MMI':
    image_loss_func = losses.MultiScaleMINDCosine()
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)


dice_loss_func = losses.Dice().loss

if bidir:
    Losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    Losses = [image_loss_func]
    weights = [0.5]
Losses += [losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]


dice_history = []
x_history = []
train_loss_history = []
val_loss_history = []
for epoch in range(args.initial_epoch, args.epochs):

    if epoch % 1 == 0:
        model.save(os.path.join(model_dir, '%04d.pt' % epoch))

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []
    for step in range(args.steps_per_epoch):

        step_start_time = time.time()
        inputs, y_true, masks, _ = next(generator)
        inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
        y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
        masks = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in masks]
        extra_roi = [masks, masks[0]]

        y_pred = model(*inputs)
        flow_field = y_pred[1]

        loss = 0
        loss_list = []

        for n, loss_function in enumerate(Losses):
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss


        warped_moving_mask = stn(masks[0], flow_field)


        dice_loss_value = dice_loss_func(masks[1], warped_moving_mask) * args.dice_weight
        loss += dice_loss_value
        loss_list.append(dice_loss_value.item())

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        epoch_step_time.append(time.time() - step_start_time)

    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

    avg_train_loss = np.mean(epoch_total_loss)
    train_loss_history.append(avg_train_loss)

#######################################
    # Validation after each epoch
    model.eval()  # Set model to evaluation mode

    all_results = []
    val_losses = []
    val_dice_scores = []
    val_tre_scores = []
    val_x_scores = []
    loss_list_val = []
    tre0_list = []
    with torch.no_grad():
        for _ in range(len(os.listdir(val_fixed_images_dir))):
            val_inputs, val_y_true, val_masks, val_eval = next(val_generator)
            val_inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in val_inputs]
            val_y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in val_y_true]
            val_masks = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in val_masks]
            val_eval = [torch.from_numpy(d).to(device).float().permute(0, 5, 1, 2, 3, 4) for d in val_eval]
            val_y_pred = model(*val_inputs)
            val_flow_field = val_y_pred[1]
            val_loss = 0
            for n, loss_function in enumerate(Losses):
                curr_loss = loss_function(val_y_true[n], val_y_pred[n]) * weights[n]
                loss_list_val.append(curr_loss.item())
                val_loss += curr_loss

            val_warped_moving_mask = stn(val_masks[0], val_flow_field)

            dice_loss_value_val = dice_loss_func(val_masks[1], val_warped_moving_mask) * args.dice_weight
            val_loss += dice_loss_value_val
            loss_list_val.append(dice_loss_value_val.item())

            dice_score = dice_coefficient(val_masks[1].cpu().numpy(), val_warped_moving_mask.cpu().numpy())
            val_dice_scores.append(dice_score.item())
            fixed_labels = val_eval[1]

            wrap_labels = val_eval[0]
            val_losses.append(val_loss.item())
    avg_val_loss = np.mean(val_losses)
    avg_dice_score = np.mean(val_dice_scores)
    dice_history.append(avg_dice_score)
    val_loss_history.append(avg_val_loss)

    best_dice, best_dice_index = max((value, index) for index, value in enumerate(dice_history))

    print('Validation - loss: %.4e  DICE: %.4f' % (avg_val_loss, avg_dice_score))
    print('Validation - Best DICE epochs: %d' % (best_dice_index))
    print('Validation - Best DICE: %.4f' % (best_dice))

    model.train()

model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))