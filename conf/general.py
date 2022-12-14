
CMAP_PREFIX = 'cmap'
LABEL_PREFIX = 'label'
OPT_PREFIX = 'opt'
SAR_PREFIX = 'sar'
TRAIN_PREFIX = 'train'
VAL_PREFIX = 'val'
PREVIOUS_PREFIX = 'prev'
PREDICTION_PREFIX = 'pred'

PATCH_SIZE = 128
OVERLAP_PROP = 0.7

N_IMAGES_YEAR = 3

N_CLASSES = 3
N_OPTICAL_BANDS = 13
N_SAR_BANDS = 2

EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 0.00005
EARLY_STOP_MIN_EPOCHS = 0
MAX_EPOCHS = 500

LEARNING_RATE = 5e-5

LEARNING_RATE_SCHEDULER_GAMMA = 1

CLASSES_WEIGHTS = [0.2, 0.8, 0.0]

#PREDICTION_OVERLAPS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.7]
PREDICTION_OVERLAPS = [0.15, 0.2, 0.25, 0.3]
PREDICTION_REMOVE_BORDER = 8 #pixels removed from each predicted patch border

MIN_CLOUD_COVER = 70