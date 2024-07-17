import torch
import numpy as np
from models.fins.model import FilteredNoiseShaper
from models.fins.utils.audio import crop_rir
from models.model_utils import get_direct_window, find_direct_location


def generate(rir, config) : 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model

    ##########################################################
    model_file = "./models/fins/checkpoints/epoch-749.pt"
    ##########################################################

    model = FilteredNoiseShaper(config.model.params)
    state_dicts = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dicts["model_state_dict"])
    model = model.to(device)
    model.eval()

    
    rir_length = int(config.dataset.params.rir_duration * config.sample_rate)
    noise_condition_length = config.model.params.noise_condition_length

    rir = rir[np.newaxis, :]
    
    cropped_rir = crop_rir(rir, target_length=rir_length)
    norm_factor = np.max(np.abs(cropped_rir))
    cropped_rir = (cropped_rir / norm_factor) * 0.999
    cropped_rir = torch.FloatTensor(cropped_rir).to(device)
    cropped_rir = torch.unsqueeze(cropped_rir, 1)
    print (cropped_rir.shape)

    stochastic_noise = torch.randn((1, 1, rir_length), device=device)
    stochastic_noise = stochastic_noise.repeat(1, config.model.params.num_filters, 1)
    noise_condition = torch.randn((1, noise_condition_length), device=device)

    # PREDICT 
    pred_rir = model(cropped_rir, stochastic_noise, noise_condition)

    pred_rir = pred_rir * norm_factor 

    pred_rir = pred_rir[0][0].detach().cpu().numpy() # 1 D 

    rir = rir[0]

    if len(pred_rir) > len(rir):
        pred_rir = pred_rir[:len(rir)]
    else : 
        _pred_rir = np.zeros_like(rir)
        _pred_rir[:len(pred_rir)] = pred_rir
        pred_rir = _pred_rir
    

    # Shift to match direct location
    original_direct_location = find_direct_location(rir)
    fins_direct_location = 0
    
    shift_amount = original_direct_location - fins_direct_location
    pred_rir_shifted = np.zeros_like(rir)
    if shift_amount < 0 : 
        # need to shift left 
        print ("FINS is more delayed than original")
        pred_rir_shifted[:len(pred_rir) + shift_amount] = pred_rir[abs(shift_amount):]
    else :
        # need to shift right 
        print ("FINS is earlier than original")
        # pred_rir_shifted[shift_amount:shift_amount + ] = pred_rir[:len(rir) - shift_amount]
        pred_rir_shifted[shift_amount:shift_amount + len(pred_rir)] = pred_rir[:-shift_amount]

    pred_rir_shifted_full_length = np.zeros_like(rir)
    pred_rir_shifted_full_length[:len(pred_rir_shifted)] = pred_rir_shifted
    

    # Replace direct part 
    original_direct_window, original_direct_length = get_direct_window(rir, config.sample_rate)
    original_direct = original_direct_window * rir

    fins_direct_window, _ = get_direct_window(pred_rir_shifted_full_length, config.sample_rate)
    fins_direct = fins_direct_window * pred_rir_shifted_full_length 

    pred_rir_shifted_full_length -= fins_direct
    pred_rir_shifted_full_length += original_direct 
    return pred_rir_shifted_full_length
