from conf import default, general, paths
from models.models import ResUnetOpt

def get_model():
    print('Model Optical Resunet')
    lidar_bands = None
    input_depth_0 = 2*general.N_OPTICAL_BANDS + 3
    input_depth_1 = 0
    model_depths = [32, 64, 128, 256]
    print(f'Model size: {model_depths}')
    print(f'Input depths: {input_depth_0}, {input_depth_1}')
    model = ResUnetOpt(input_depth_0, model_depths, general.N_CLASSES)

    return model, lidar_bands