import argparse
import pathlib
from conf import paths, general, default
import numpy as np
import os
from utils.ops import load_opt_image, load_SAR_image, load_single_band_image
from skimage.util import view_as_windows
import sys
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='prepare the original files, generating .npy files to be used in the training/testing steps'
)

parser.add_argument( # Path to the optical images
    '--opt-path',
    type = pathlib.Path,
    default = paths.OPT_PATH,
    help = 'Path to the optical images'
)

parser.add_argument( # Path to the SAR images
    '--sar-path',
    type = pathlib.Path,
    default = paths.SAR_PATH,
    help = 'Path to the SAR images'
)

parser.add_argument( # List of optical images
    '--opt-imgs',
    type = str,
    default = paths.OPT_IMGS,
    nargs='+',
    help = 'List of optical images'
)

parser.add_argument( # List of SAR images
    '--sar-imgs',
    type = str,
    default = paths.SAR_IMGS,
    nargs='+',
    help = 'List of SAR images'
)

parser.add_argument( # List of years to be processed
    '-y', '--years',
    type = int,
    default = default.YEARS,
    nargs='+',
    help = 'List of years to be prepared'
)

parser.add_argument( # The path to the experiments' folder
    '-x', '--experiments-folder',
    type = pathlib.Path,
    default = paths.EXPERIMENTS_FOLDER,
    help = 'The path to the experiments\' folder'
)

parser.add_argument( # The path to the general data (labels, cloud maps and tiles) folder
    '-l', '--general-folder',
    type = pathlib.Path,
    default = paths.GENERAL_PATH,
    help = 'The path to the general data (labels, cloud maps and tiles) folder'
)

parser.add_argument( # The minimum proportion of deforestation labels in each train/validation patches
    '-d', '--def-min-prop',
    type =float,
    default = default.DEF_CLASS_MIN_PROP,
    help = 'The minimum proportion of deforestation labels in each train/validation patches'
)

args = parser.parse_args()

if not os.path.exists(paths.PREPARED_PATH):
    os.mkdir(paths.PREPARED_PATH)

np.random.seed(123)

outfile = os.path.join(args.experiments_folder, 'data-prep.txt')
with open(outfile, 'w') as sys.stdout:
    print('Generating statistics')
    opt_means, opt_stds = [], []
    for opt_img in tqdm(args.opt_imgs, desc = 'Opening Optical'):
        img = load_opt_image(os.path.join(args.opt_path, opt_img))
        img[np.isnan(img)] = 0
        opt_means.append(img.mean(axis=(0,1)))
        opt_stds.append(img.std(axis=(0,1)))

    opt_mean = np.array(opt_means).mean(axis=0)
    opt_std = np.array(opt_stds).mean(axis=0)

    print(f'Optical means: {opt_mean}')
    print(f'Optical stds: {opt_std}')

    sar_means, sar_stds = [], []
    for sar_img in tqdm(args.sar_imgs, desc = 'Opening SAR'):
        img = load_SAR_image(os.path.join(args.sar_path, sar_img))
        img[np.isnan(img)] = 0
        sar_means.append(img.mean(axis=(0,1)))
        sar_stds.append(img.std(axis=(0,1)))

    sar_mean = np.array(sar_means).mean(axis=0)
    sar_std = np.array(sar_stds).mean(axis=0)

    print(f'SAR means: {sar_mean}')
    print(f'SAR stds: {sar_std}')

    print('Preparing normalized data')
    for opt_img in tqdm(args.opt_imgs, desc = 'Preparing Optical'):
        img = load_opt_image(os.path.join(args.opt_path, opt_img))
        img[np.isnan(img)] = 0
        img = (img - opt_mean)/opt_std
        print(f'Optical Image {opt_img} means: {img.mean(axis=(0,1))}')
        print(f'Optical Image {opt_img} stds: {img.std(axis=(0,1))}')
        np.save(os.path.join(paths.PREPARED_PATH, f'{general.OPT_PREFIX}_{opt_img[:-4]}'), img.astype(np.float16))

    for sar_img in tqdm(args.sar_imgs, desc = 'Preparing SAR'):
        img = load_SAR_image(os.path.join(args.sar_path, sar_img))
        img[np.isnan(img)] = 0
        img = (img - sar_mean)/sar_std
        print(f'SAR Image {sar_img} means: {img.mean(axis=(0,1))}')
        print(f'SAR Image {sar_img} stds: {img.std(axis=(0,1))}')
        np.save(os.path.join(paths.PREPARED_PATH, f'{general.SAR_PREFIX}_{sar_img[:-4]}'), img.astype(np.float32))

    print('Preparing general data')
    print('Preparing Cloud Maps')
    for opt_img in tqdm(args.opt_imgs, desc = 'Preparing Cloud Maps'):
        cmap = (load_single_band_image(os.path.join(args.general_folder, f'{general.CMAP_PREFIX}_{opt_img}'))).astype(np.float16)
        cmap = 100*cmap
        cmap = cmap.round().astype(np.uint8)
        np.save(os.path.join(paths.PREPARED_PATH, f'{general.CMAP_PREFIX}_{opt_img[:-4]}'), cmap)
        

    print('Preparing Labels')
    for year in tqdm(args.years[1:], desc = 'Preparing Labels'):
        label = load_single_band_image(os.path.join(args.general_folder, f'{general.LABEL_PREFIX}_{year}.tif')).astype(np.uint8)
        np.save(os.path.join(paths.PREPARED_PATH, f'{general.LABEL_PREFIX}_{year}'), label)

    print('Preparing Previous Def maps')
    for year in tqdm(args.years[1:], desc = 'Preparing Previous Def Map'):
        previous = load_single_band_image(os.path.join(args.general_folder, f'{general.PREVIOUS_PREFIX}_{year}.tif')).astype(np.float16)
        np.save(os.path.join(paths.PREPARED_PATH, f'{general.PREVIOUS_PREFIX}_{year}'), previous)


    print('Preparing patches')
    tiles = load_single_band_image(paths.TILES_PATH).astype(np.uint8)
    shape = tiles.shape
    tiles = tiles.flatten()
    idx = np.arange(shape[0] * shape[1]).reshape(shape)
    window_shape = (general.PATCH_SIZE, general.PATCH_SIZE)
    slide_step = int((1-general.OVERLAP_PROP)*general.PATCH_SIZE)
    idx_patches = view_as_windows(idx, window_shape, slide_step).reshape((-1, general.PATCH_SIZE, general.PATCH_SIZE))
    min_prop = args.def_min_prop


    for year in tqdm(args.years[1:-1], desc = 'Preparing patches'):
        label = load_single_band_image(os.path.join(args.general_folder, f'{general.LABEL_PREFIX}_{year}.tif')).astype(np.uint8).flatten()

        keep = ((label[idx_patches] == 1).sum(axis=(1,2)) / general.PATCH_SIZE**2) >= min_prop

        keep_args = np.argwhere(keep == True).flatten() #args with at least min_prop deforestation
        no_keep_args = np.argwhere(keep == False).flatten() #args with less than min_prop of deforestation
        no_keep_args = np.random.choice(no_keep_args, (keep==True).sum())

        keep_final = np.concatenate((keep_args, no_keep_args))

        all_idx_patches = idx_patches[keep_final]
        #all_idx_patches = idx_patches[keep_args]

        keep_val = (tiles[all_idx_patches] == 0).sum(axis=(1,2)) == general.PATCH_SIZE**2
        keep_train = (tiles[all_idx_patches] == 1).sum(axis=(1,2)) == general.PATCH_SIZE**2

        print(f'Train patches: {keep_train.sum()}')
        print(f'Validation patches: {keep_val.sum()}')

        val_idx_patches = all_idx_patches[keep_val]
        train_idx_patches = all_idx_patches[keep_train]

        np.save(os.path.join(paths.PREPARED_PATH, f'{general.VAL_PREFIX}_{year}'), val_idx_patches)
        np.save(os.path.join(paths.PREPARED_PATH, f'{general.TRAIN_PREFIX}_{year}'), train_idx_patches)




