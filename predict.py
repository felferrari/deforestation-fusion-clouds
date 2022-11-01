import argparse
import pathlib
import importlib
from conf import default, general, paths
import os
import time
import sys
from utils.dataloader import PredDataSet
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
from osgeo import ogr, gdal, gdalconst
from utils.ops import load_dict, save_geotiff
from multiprocessing import Process

parser = argparse.ArgumentParser(
    description='Train NUMBER_MODELS models based in the same parameters'
)

parser.add_argument( # Experiment number
    '-e', '--experiment',
    type = int,
    default = 1,
    help = 'The number of the experiment'
)

parser.add_argument( # batch size
    '-b', '--batch-size',
    type = int,
    default = default.PREDICTION_BATCH_SIZE,
    help = 'The number of samples of each batch'
)

parser.add_argument( # Number of models to be trained
    '-n', '--number-models',
    type = int,
    default = default.N_MODELS,
    help = 'The number models to be trained from the scratch'
)

parser.add_argument( # Experiment path
    '-x', '--experiments-path',
    type = pathlib.Path,
    default = paths.EXPERIMENTS_FOLDER,
    help = 'The patch to data generated by all experiments'
)

parser.add_argument( # Reference year to predict
    '-y', '--year',
    type = int,
    default = default.YEARS[2],
    help = 'Reference year to predict'
)

parser.add_argument( # Base image to generate geotiff pred
    '-i', '--base-image',
    type = pathlib.Path,
    default = paths.OPT_TIFF_FILE_0,
    help = 'The patch to base image to generate Geotiff prediction'
)

args = parser.parse_args()

exp_path = os.path.join(str(args.experiments_path), f'exp_{args.experiment}')
logs_path = os.path.join(exp_path, f'logs')
models_path = os.path.join(exp_path, f'models')
visual_path = os.path.join(exp_path, f'visual')
predicted_path = os.path.join(exp_path, f'predicted')
results_path = os.path.join(exp_path, f'results')


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#def run(model_idx):
for model_idx in tqdm(range(args.number_models)):

    outfile = os.path.join(logs_path, f'pred_{args.experiment}_{model_idx}.txt')
    with open(outfile, 'w') as sys.stdout:

        #torch.set_num_threads(9)

        model_m =importlib.import_module(f'conf.exp_{args.experiment}')
        model = model_m.get_model()
        model.to(device)
        print(model)

        model_path = os.path.join(models_path, f'model_{model_idx}.pth')
        model.load_state_dict(torch.load(model_path))

        torch.set_num_threads(9)

        overlaps = general.PREDICTION_OVERLAPS

        for im_0 in tqdm(range(general.N_IMAGES_YEAR), leave=False):
            for im_1 in tqdm(range(general.N_IMAGES_YEAR), leave=False):
                img_pair = (im_0, im_1)
                dataset = PredDataSet(device = device, year = args.year, img_pair = img_pair)
                dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
                label = dataset.label
                print(f'Optical Image Year 0:{dataset.opt_file_0}')
                print(f'Optical Image Year 1:{dataset.opt_file_1}')
                print(f'SAR Image Year 0:{dataset.sar_file_0}')
                print(f'SAR Image Year 1:{dataset.sar_file_1}')
                print(f'CMAP Image Year 0:{dataset.cmap_file_0}')
                print(f'CMAP Image Year 1:{dataset.cmap_file_1}')
                print(f'Prev Def Image Year 1:{dataset.prev_def_file}')
                pred_global_sum = np.zeros(dataset.original_shape+(general.N_CLASSES,))
                t0 = time.perf_counter()
                
                # pbar = tqdm(dataloader)
                preds = None
                for X in tqdm(dataloader, leave = False):
                    with torch.no_grad():
                        pred = model(X).to('cpu')[:, :, general.PREDICTION_REMOVE_BORDER:-general.PREDICTION_REMOVE_BORDER, general.PREDICTION_REMOVE_BORDER:-general.PREDICTION_REMOVE_BORDER]
                    if preds is None:
                        preds = pred#.to('cpu')
                    else:
                        preds = torch.cat((preds, pred), dim=0)
                preds = preds.view((dataset.patches_shape)+preds.shape[1:]).numpy().astype(np.float16)
                preds = np.moveaxis(preds, 2, -1)
                pred_padded = None
                for pred_line in tqdm(preds, leave = False):
                    pred_list = [pred_i for pred_i in pred_line]
                    pred_line_f = np.column_stack(pred_list)
                    if pred_padded is  None:
                        pred_padded = pred_line_f
                    else:
                        pred_padded = np.row_stack([pred_padded, pred_line_f])
                pred_final = pred_padded[:general.PREDICTION_REMOVE_BORDER-dataset.pad_0, :general.PREDICTION_REMOVE_BORDER-dataset.pad_1]

                np.save(os.path.join(predicted_path, f'{general.PREDICTION_PREFIX}_prob_{img_pair[0]}_{img_pair[1]}_{model_idx}.npy'), pred_final[:,:,1].astype(np.float16))

                #save_geotiff(str(args.base_image), os.path.join(visual_path, f'{general.PREDICTION_PREFIX}_{args.experiment}_{img_pair[0]}_{img_pair[1]}_{model_idx}.tif'), pred_b, dtype = 'byte')

                pred_b2 = (pred_final[:,:,1] > 0.5).astype(np.uint8)
                pred_b2[label == 2] = 2
                #save_geotiff(str(args.base_image), os.path.join(visual_path, f'{general.PREDICTION_PREFIX}_{args.experiment}_{img_pair[0]}_{img_pair[1]}_{model_idx}.tif'), pred_b2, dtype = 'byte')
                #save_geotiff(str(args.base_image), os.path.join(visual_path, f'{general.PREDICTION_PREFIX}_probs_{args.experiment}_{img_pair[0]}_{img_pair[1]}_{model_idx}.tif'), pred_final[:,:,1], dtype = 'float')
