from conf import default, general, paths
from models.models import JointFusionNoSkip

def get_model():
    print('Model Joint fusion without skipp connections')
    input_depth_0 = 2*general.N_OPTICAL_BANDS + 1
    input_depth_1 = 2*general.N_SAR_BANDS + 1
    model_depths = [32, 64, 128, 256]
    print(f'Model size: {model_depths}')
    print(f'Input depths: {input_depth_0}, {input_depth_1}')
    model = JointFusionNoSkip(input_depth_0, input_depth_1, model_depths, general.N_CLASSES)

    return model