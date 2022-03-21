import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from tqdm import tqdm


""" Script to compute the CW-SSIM """


def get_cw_ssim(input, target):
    """
    Computes the CW SSIM
    
    Params:
        input (np.ndarray): shape [n_time, n_lat, n_lon]
        target (np.ndarray): shape [n_time, n_lat, n_lon]
        
    Returns:
        mean (float):
            time averaged CW SSIM
    """
    from ssim import SSIM
    from PIL import Image
    
    mean = 0
    for i in tqdm(range(len(input))):
        data_1 = input[i]
        data_2 = target[i]

        data_normed_1 = Normalize(data_1.min(), data_1.max())(data_1)
        data_normed_2 = Normalize(data_2.min(), data_2.max())(data_2)

        im_1 = Image.fromarray(np.uint8(cm.gray(data_normed_1)*255)) 
        im_2 = Image.fromarray(np.uint8(cm.gray(data_normed_2)*255)) 

        cw_ssim = SSIM(im_1).cw_ssim_value(im_2)
        mean += cw_ssim
    mean /= len(input)
    return mean 


def main():

    model_name = 'dnn.npy'
    in_path ='/path/to/models'

    tmp = np.load(f'{in_path}/{model_name}')
    model, target = tmp[0], tmp[1]

    print(f'Computing the CW-SSIM from {in_path}/{model_name}:')
    result = get_cw_ssim(model, target)
    print(f'Result: {result}')

if __name__ == '__main__':
    main()