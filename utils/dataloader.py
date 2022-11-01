from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.nn.functional import one_hot
import torch
import numpy as np
import os
from conf import paths, general
import random
from torchvision.transforms.functional import hflip, vflip
from skimage.util import view_as_windows, view_as_blocks

class TrainDataSet(Dataset):
    def __init__(self, ds, device, year, data_aug = False, transformer = ToTensor()) -> None:
        
        self.device = device
        self.data_aug = data_aug
        self.transformer = transformer

        self.year_0 = str(year-1)[2:]
        self.year_1 = str(year)[2:]

        prep_files = os.listdir(paths.PREPARED_PATH)
        path_to_idx_patches = os.path.join(paths.PREPARED_PATH, f'{ds}_{year}.npy')
        self.idx_patches = np.load(path_to_idx_patches)
        np.random.seed(123)
        np.random.shuffle(self.idx_patches)

        opt_files_0 = [os.path.join(paths.PREPARED_PATH, fi) for fi in prep_files if fi.startswith(f'{general.OPT_PREFIX}_{self.year_0}')]
        self.opt_imgs_0 = []
        for opt_file_0 in opt_files_0:
            img = np.load(opt_file_0)
            self.opt_imgs_0.append(img.reshape((-1, img.shape[-1])))

        opt_files_1 = [os.path.join(paths.PREPARED_PATH, fi) for fi in prep_files if fi.startswith(f'{general.OPT_PREFIX}_{self.year_1}')]
        self.opt_imgs_1 = []
        for opt_file_1 in opt_files_1:
            img = np.load(opt_file_1)
            self.opt_imgs_1.append(img.reshape((-1, img.shape[-1])))

        sar_files_0 = [os.path.join(paths.PREPARED_PATH, fi) for fi in prep_files if fi.startswith(f'{general.SAR_PREFIX}_{self.year_0}')]
        self.sar_imgs_0 = []
        for sar_file_0 in sar_files_0:
            img = np.load(sar_file_0)
            self.sar_imgs_0.append(img.reshape((-1, img.shape[-1])))

        sar_files_1 = [os.path.join(paths.PREPARED_PATH, fi) for fi in prep_files if fi.startswith(f'{general.SAR_PREFIX}_{self.year_1}')]
        self.sar_imgs_1 = []
        for sar_file_1 in sar_files_1:
            img = np.load(sar_file_1)
            self.sar_imgs_1.append(img.reshape((-1, img.shape[-1])))

        cmap_files_0 = [os.path.join(paths.PREPARED_PATH, f'{general.CMAP_PREFIX}_{fi[13:]}') for fi in opt_files_0]
        self.cmaps_0 = []
        for cmap_file_0 in cmap_files_0:
            img = np.load(cmap_file_0)
            self.cmaps_0.append(img.reshape((-1, 1))/100)

        cmap_files_1 = [os.path.join(paths.PREPARED_PATH, f'{general.CMAP_PREFIX}_{fi[13:]}') for fi in opt_files_1]
        self.cmaps_1 = []
        for cmap_file_1 in cmap_files_1:
            img = np.load(cmap_file_1)
            self.cmaps_1.append(img.reshape((-1, 1))/100)

        label_file = os.path.join(paths.PREPARED_PATH, f'{general.LABEL_PREFIX}_{year}.npy')
        self.label = np.load(label_file)
        self.label = self.label.flatten()

        prev_def_file = os.path.join(paths.PREPARED_PATH, f'{general.PREVIOUS_PREFIX}_{year}.npy')
        self.prev_def = np.load(prev_def_file)
        self.prev_def = self.prev_def.reshape((-1, 1))


    def __len__(self):
        return self.idx_patches.shape[0] * general.N_IMAGES_YEAR**2

    def __getitem__(self, index):
        idx_patch = index // (general.N_IMAGES_YEAR**2)
        im_0 = (index % general.N_IMAGES_YEAR) // general.N_IMAGES_YEAR
        im_1 = (index % general.N_IMAGES_YEAR) %  general.N_IMAGES_YEAR

        patch = self.idx_patches[idx_patch]

        if self.data_aug:
            k = random.randint(0, 3)
            patch = np.rot90(patch)

            if bool(random.getrandbits(1)):
                patch = np.flip(patch, axis=0)

            if bool(random.getrandbits(1)):
                patch = np.flip(patch, axis=1)

        opt_0 = self.transformer(self.opt_imgs_0[im_0][patch].astype(np.float32)).to(self.device)
        opt_1 = self.transformer(self.opt_imgs_1[im_1][patch].astype(np.float32)).to(self.device)

        sar_0 = self.transformer(self.sar_imgs_0[im_0][patch].astype(np.float32)).to(self.device)
        sar_1 = self.transformer(self.sar_imgs_1[im_1][patch].astype(np.float32)).to(self.device)

        cmap_0 = self.transformer(self.cmaps_0[im_0][patch].astype(np.float32)).to(self.device)
        cmap_1 = self.transformer(self.cmaps_1[im_1][patch].astype(np.float32)).to(self.device)

        prev_def = self.transformer(self.prev_def[patch].astype(np.float32)).to(self.device)
        label = torch.tensor(self.label[patch].astype(np.int64)).to(self.device)

        return (
            opt_0,
            opt_1,
            sar_0,
            sar_1,
            cmap_0,
            cmap_1,
            prev_def
        ), label


class PredDataSet(Dataset):
    def __init__(self, device, year, img_pair, transformer = ToTensor()) -> None:
        self.device = device
        self.transformer = transformer

        self.year_0 = str(year-1)[2:]
        self.year_1 = str(year)[2:]

        prep_files = os.listdir(paths.PREPARED_PATH)

        label_file = os.path.join(paths.PREPARED_PATH, f'{general.LABEL_PREFIX}_{year}.npy')
        self.label = np.load(label_file)
        self.original_shape = self.label.shape
        pred_step = general.PATCH_SIZE - 2 * general.PREDICTION_REMOVE_BORDER

        self.pad_0 = pred_step - (self.original_shape[0] % pred_step) + general.PREDICTION_REMOVE_BORDER
        self.pad_1 = pred_step - (self.original_shape[1] % pred_step) + general.PREDICTION_REMOVE_BORDER

        pad_shape = ((general.PREDICTION_REMOVE_BORDER, self.pad_0), (general.PREDICTION_REMOVE_BORDER, self.pad_1), (0, 0))

        opt_files_0 = [os.path.join(paths.PREPARED_PATH, fi) for fi in prep_files if fi.startswith(f'{general.OPT_PREFIX}_{self.year_0}')]
        self.opt_file_0 = opt_files_0[img_pair[0]]
        img = np.load(self.opt_file_0)
        img = np.pad(img, pad_shape, mode = 'reflect')
        self.opt_img_0 = img.reshape((-1, img.shape[-1]))

        opt_files_1 = [os.path.join(paths.PREPARED_PATH, fi) for fi in prep_files if fi.startswith(f'{general.OPT_PREFIX}_{self.year_1}')]
        self.opt_file_1 = opt_files_1[img_pair[1]]
        img = np.load(self.opt_file_1)
        img = np.pad(img, pad_shape, mode = 'reflect')
        self.opt_img_1 = img.reshape((-1, img.shape[-1]))


        sar_files_0 = [os.path.join(paths.PREPARED_PATH, fi) for fi in prep_files if fi.startswith(f'{general.SAR_PREFIX}_{self.year_0}')]
        self.sar_file_0 = sar_files_0[img_pair[0]]
        img = np.load(self.sar_file_0)
        img = np.pad(img, pad_shape, mode = 'reflect')
        self.sar_img_0 = img.reshape((-1, img.shape[-1]))


        sar_files_1 = [os.path.join(paths.PREPARED_PATH, fi) for fi in prep_files if fi.startswith(f'{general.SAR_PREFIX}_{self.year_1}')]
        self.sar_file_1 = sar_files_1[img_pair[1]]
        img = np.load(self.sar_file_1)
        img = np.pad(img, pad_shape, mode = 'reflect')
        self.sar_img_1 = img.reshape((-1, img.shape[-1]))
        

        pad_shape = ((general.PREDICTION_REMOVE_BORDER, self.pad_0), (general.PREDICTION_REMOVE_BORDER, self.pad_1))
        cmap_files_0 = [os.path.join(paths.PREPARED_PATH, f'{general.CMAP_PREFIX}_{fi[13:]}') for fi in opt_files_0]
        self.cmap_file_0 = cmap_files_0[img_pair[0]]
        img = np.load(self.cmap_file_0)
        img = np.pad(img, pad_shape, mode = 'reflect')
        self.cmap_img_0 = img.reshape((-1, 1))


        cmap_files_1 = [os.path.join(paths.PREPARED_PATH, f'{general.CMAP_PREFIX}_{fi[13:]}') for fi in opt_files_1]
        self.cmap_file_1 = cmap_files_1[img_pair[1]]
        img = np.load(self.cmap_file_1)
        img = np.pad(img, pad_shape, mode = 'reflect')
        self.cmap_img_1 = img.reshape((-1, 1))


        self.prev_def_file = os.path.join(paths.PREPARED_PATH, f'{general.PREVIOUS_PREFIX}_{year}.npy')
        self.prev_def = np.load(self.prev_def_file)
        self.prev_def = np.pad(self.prev_def, pad_shape, mode = 'reflect')
        shape_padded = self.prev_def.shape[:2]
        self.prev_def = self.prev_def.reshape((-1, 1))

        idx_patches = np.arange(shape_padded[0]*shape_padded[1]).reshape(shape_padded)
        window_shape = (general.PATCH_SIZE, general.PATCH_SIZE)
        self.idx_patches = view_as_windows(idx_patches, window_shape, pred_step)#.reshape((-1, general.PATCH_SIZE, general.PATCH_SIZE))
        self.patches_shape = self.idx_patches.shape[:2]
        self.idx_patches = self.idx_patches.reshape((-1, general.PATCH_SIZE, general.PATCH_SIZE))


    def __len__(self):
        return self.idx_patches.shape[0]

    def __getitem__(self, index):
        patch = self.idx_patches[index]

        opt_0 = self.transformer(self.opt_img_0[patch].astype(np.float32)).to(self.device)
        opt_1 = self.transformer(self.opt_img_1[patch].astype(np.float32)).to(self.device)

        sar_0 = self.transformer(self.sar_img_0[patch].astype(np.float32)).to(self.device)
        sar_1 = self.transformer(self.sar_img_1[patch].astype(np.float32)).to(self.device)

        cmap_0 = self.transformer(self.cmap_img_0[patch].astype(np.float32)).to(self.device)
        cmap_1 = self.transformer(self.cmap_img_1[patch].astype(np.float32)).to(self.device)

        prev_def = self.transformer(self.prev_def[patch].astype(np.float32)).to(self.device)

        return (
            opt_0,
            opt_1,
            sar_0,
            sar_1,
            cmap_0,
            cmap_1,
            prev_def
        )