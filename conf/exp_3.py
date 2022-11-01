from conf import default, general, paths
from models.models import EarlyFusion

def get_model():
    print('Model Early Fusion sem cmap')
    input_depth_0 = 2*general.N_OPTICAL_BANDS + 2*general.N_SAR_BANDS + 1
    model_depths = [32, 64, 128, 256]
    print(f'Model size: {model_depths}')
    print(f'Input depth: {input_depth_0}')
    model = EarlyFusion(input_depth_0, model_depths, general.N_CLASSES)

    return model