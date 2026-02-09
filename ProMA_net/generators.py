import os
import glob
import numpy as np
from . import utils

def get_dataset_paths(data_dir):
    train_fixed_images_dir = os.path.join(data_dir, 'train', 'fixed_images')
    train_moving_images_dir = os.path.join(data_dir, 'train', 'moving_images')
    train_fixed_labels_dir = os.path.join(data_dir, 'train', 'fixed_labels')
    train_moving_labels_dir = os.path.join(data_dir, 'train', 'moving_labels')

    val_fixed_images_dir = os.path.join(data_dir, 'val', 'fixed_images')
    val_moving_images_dir = os.path.join(data_dir, 'val', 'moving_images')
    val_fixed_labels_dir = os.path.join(data_dir, 'val', 'fixed_labels')
    val_moving_labels_dir = os.path.join(data_dir, 'val', 'moving_labels')

    test_fixed_images_dir = os.path.join(data_dir, 'test', 'fixed_images')
    test_moving_images_dir = os.path.join(data_dir, 'test', 'moving_images')
    test_fixed_labels_dir = os.path.join(data_dir, 'test', 'fixed_labels')
    test_moving_labels_dir = os.path.join(data_dir, 'test', 'moving_labels')

    return (train_fixed_images_dir, train_moving_images_dir, train_fixed_labels_dir, train_moving_labels_dir,
            val_fixed_images_dir, val_moving_images_dir, val_fixed_labels_dir, val_moving_labels_dir,
            test_fixed_images_dir, test_moving_images_dir, test_fixed_labels_dir, test_moving_labels_dir)


def paired_scan_generator(
    fixed_image_dir,
    moving_image_dir,
    fixed_label_dir,
    moving_label_dir,
    batch_size=1,
    add_feat_axis=True
):

    fixed_image_files = sorted(glob.glob(os.path.join(fixed_image_dir, '*.nii.gz')))
    moving_image_files = sorted(glob.glob(os.path.join(moving_image_dir, '*.nii.gz')))
    fixed_label_files = sorted(glob.glob(os.path.join(fixed_label_dir, '*.nii.gz')))
    moving_label_files = sorted(glob.glob(os.path.join(moving_label_dir, '*.nii.gz')))

    assert len(fixed_image_files) == len(moving_image_files) == len(fixed_label_files) == len(moving_label_files), \
        "The number of fixed and moving images and labels must be consistent."

    # 获取数据总数
    num_samples = len(fixed_image_files)

    # 定义加载参数
    load_params = dict(
        add_batch_axis=True,
        add_feat_axis=add_feat_axis,
    )


    start_idx = 0

    while True:

        end_idx = start_idx + batch_size
        if end_idx > num_samples:

            start_idx = 0
            end_idx = batch_size


        indices = list(range(start_idx, end_idx))

        fixed_images = []
        moving_images = []
        fixed_labels = []
        moving_labels = []

        for idx in indices:

            fixed_image = utils.load_volfile(fixed_image_files[idx], **load_params)
            moving_image = utils.load_volfile(moving_image_files[idx], **load_params)


            fixed_label = utils.load_volfile(fixed_label_files[idx], **load_params)
            moving_label = utils.load_volfile(moving_label_files[idx], **load_params)

            fixed_images.append(fixed_image)
            moving_images.append(moving_image)
            fixed_labels.append(fixed_label)
            moving_labels.append(moving_label)


        fixed_images = np.concatenate(fixed_images, axis=0)
        moving_images = np.concatenate(moving_images, axis=0)
        fixed_labels = np.concatenate(fixed_labels, axis=0)
        moving_labels = np.concatenate(moving_labels, axis=0)


        inputs = [moving_images, fixed_images]
        y_true = [fixed_images, moving_images]
        masks_train = [moving_labels[:, :, :, :, 0, :], fixed_labels[:, :, :, :, 0, :]]
        masks_val = [moving_labels, fixed_labels]


        start_idx = end_idx

        yield (inputs, y_true, masks_train, masks_val)
