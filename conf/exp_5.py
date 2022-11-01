from conf import default, general, paths
from models.models import LateFusion

def get_model():
    print('Model Late fusion')
    input_depth_0 = 2*general.N_OPTICAL_BANDS + 3
    input_depth_1 = 2*general.N_SAR_BANDS + 1
    model_depths = [32, 64, 128, 256]
    print(f'Model size: {model_depths}')
    print(f'Input depths: {input_depth_0}, {input_depth_1}')
    model = LateFusion(input_depth_0, input_depth_1, model_depths, general.N_CLASSES)

    return model