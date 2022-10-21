import argparse
import pathlib
from conf import paths, general
import os
from utils.ops import load_opt_image, save_geotiff
from s2cloudless import S2PixelCloudDetector
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='prepare the original files, generating .npy files to be used in the training/testing steps'
)

parser.add_argument( # Folder with the optical files
    '--opt-folder',
    type = pathlib.Path,
    default = paths.OPT_PATH,
    help = 'Path to the optical images folder'
)


parser.add_argument( # image
    '--opt-imgs',
    type = str,
    default = paths.OPT_IMGS,
    nargs='+',
    help = 'List of the optical images'
)

parser.add_argument( # output folder
    '-o', '--output-folder',
    type = pathlib.Path,
    default = paths.GENERAL_PATH,
    help = 'Path to the folder of the output cloud maps files (.tif)'
)

args = parser.parse_args()

if not os.path.exists(paths.PREPARED_PATH):
    os.mkdir(paths.PREPARED_PATH)

#outfile = f'preparation.txt'
#with open(outfile, 'w') as sys.stdout:

cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2, all_bands=True)

for opt_file_name in tqdm(args.opt_imgs):
    img = load_opt_image(os.path.join(args.opt_folder, opt_file_name))/10000
    cloud_prob = cloud_detector.get_cloud_probability_maps(img)
    save_geotiff(os.path.join(args.opt_folder, opt_file_name), os.path.join(args.output_folder, f'{general.CMAP_PREFIX}_{opt_file_name}'), cloud_prob, 'float')