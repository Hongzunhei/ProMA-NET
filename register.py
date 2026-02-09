#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register
a scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.pt
        --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.

If you use this code, please cite the following, and read function docs for further info/citations
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca.
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019.

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
implied. See the License for the specific language governing permissions and limitations under
the License.
"""

import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import argparse
# third party
import numpy as np
import torch
from ProMA_net import generators, layers, utils
import ProMA_net.ProMA_NET as ProMA_NET
# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='', help='root directory of the dataset')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--model', default="", help='pytorch model for nonlinear registration')
parser.add_argument('--output-dir', default='output', help='root directory of the dataset')
parser.add_argument('--warp', default=True, help='output warp deformation filename')
parser.add_argument('-g', '--gpu', default='3',  help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def dice_coefficient(mask1, mask2):

    mask1 = np.round(mask1)
    mask2 = np.round(mask2)


    intersection = np.logical_and(mask1, mask2).sum()
    volume_sum = mask1.sum() + mask2.sum()

    if volume_sum == 0:
        return 1.0

    return 2.0 * intersection / volume_sum


_, _, _, _,_, _, _, _,test_fixed_images_dir, test_moving_images_dir, test_fixed_labels_dir, test_moving_labels_dir  = generators.get_dataset_paths(args.data_dir)

generator = generators.paired_scan_generator(
    fixed_image_dir=test_fixed_images_dir,
    moving_image_dir=test_moving_images_dir,
    fixed_label_dir=test_fixed_labels_dir,
    moving_label_dir=test_moving_labels_dir,
    batch_size=args.batch_size,
    add_feat_axis=True
)
sample_input = next(generator)
inshape = sample_input[0][0].shape[1:-1]
stn = layers.SpatialTransformer(inshape).to(device)
moving_images = sorted(os.listdir(test_moving_images_dir))
fixed_images = sorted(os.listdir(test_fixed_images_dir))
moving_labels = sorted(os.listdir(test_fixed_images_dir))
common_files = set(moving_images).intersection(set(fixed_images))
common_files = sorted(common_files)
if len(common_files) == 0:
    print("No images with the same filename were found. Please check the input folders.")
    exit(1)
print(f"Found {len(common_files)} pairs of images with the same filename. Starting registration...")


# load moving and fixed images
add_feat_axis = not args.multichannel

inshape = sample_input[0][0].shape[1:-1]
model = ProMA_NET.ProMANET.load(args.model, device)
model.to(device)
model.eval()
stn = layers.SpatialTransformer(inshape).to(device)
dice_ = []
#######################################################
with torch.no_grad():
    for filename in common_files:
        moving_path = os.path.join(test_moving_images_dir, filename)
        fixed_path = os.path.join(test_fixed_images_dir, filename)
        fixed_labels_path = os.path.join(test_fixed_labels_dir, filename)
        moving_label_path = os.path.join(test_moving_labels_dir, filename)
        moving = utils.load_volfile(moving_path, add_batch_axis=True, add_feat_axis=add_feat_axis)
        fixed, fixed_affine = utils.load_volfile(fixed_path, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
        f_label, f_label_affine = utils.load_volfile(fixed_labels_path, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
        m_label, m_label_affine = utils.load_volfile(moving_label_path, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
        input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
        input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)
        fixed_label = torch.from_numpy(f_label).to(device).float().permute(0, 5, 1, 2, 3, 4)
        input_label = torch.from_numpy(m_label).to(device).float().permute(0, 5, 1, 2, 3, 4)
        moved, warp = model(input_moving, input_fixed)

        moved = moved.detach().cpu().numpy().squeeze()
        moved = moved[..., np.newaxis]
        moved_filename = os.path.join(args.output_dir, "moving_images", f"{filename}")
        utils.save_volfile(moved, moved_filename, fixed_affine)
        data_list = []


        for i in range(6):
            label = input_label[..., i]
            wrap_mask = stn(label, warp)
            if i == 0:
                dice_score = dice_coefficient(fixed_label[..., 0].cpu().numpy(), wrap_mask.cpu().numpy())
                print(dice_score)
                dice_.append(dice_score)
            stacked_data = wrap_mask.detach().cpu().numpy().squeeze()
            data_list.append(stacked_data)
        mask_filename = os.path.join(args.output_dir, "moving_labels", f"{filename}")
        stacked_data = np.stack(data_list, axis=-1)
        utils.save_volfile(stacked_data, mask_filename, fixed_affine)


        if args.warp:
            warp = warp.detach().cpu().numpy().squeeze()
            warp_filename = os.path.join(args.output_dir, "ddf", f"{filename}")
            utils.save_volfile(warp, warp_filename, fixed_affine)
    avg_dice_score = np.mean(dice_)
print(f"All registrations have been completed. {avg_dice_score}")



