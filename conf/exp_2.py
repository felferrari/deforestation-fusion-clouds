from conf import default, general, paths
from models.models import ResUnetSAR

def get_model():
    print('Model SAR Resunet')
    lidar_bands = None
    input_depth_0 = 2*general.N_SAR_BANDS + 1
    input_depth_1 = 0
    model_depths = [32, 64, 128, 256]
    print(f'Model size: {model_depths}')
    print(f'Input depths: {input_depth_0}, {input_depth_1}')
    model = ResUnetSAR(input_depth_0, model_depths, general.N_CLASSES)

    return model, lidar_bands