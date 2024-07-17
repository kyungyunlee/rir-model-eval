import os
import torch
import soundfile as sf 
import pandas as pd
from pathlib import Path
from models.fins.utils.audio import load_audio
from models.fins.utils.utils import load_config
from models.model_utils import process_rir
from models.fins import fins 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main() : 
    # Load config
    config_path = "./models/fins/config.yaml"
    config = load_config(config_path)

    
    output_dir = "./models/fins/result"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False) 


    version_rir_df = pd.read_csv(f"./datasets/rir_list.csv", delimiter=",")

    for i, series in version_rir_df.iterrows() : 
        path, filename, rt, real  = series
        print (path, filename, rt, real)
        rir_file = Path("./") / path / filename
        rir = load_audio(rir_file, target_sr=config.sample_rate, mono=True)
        _rir = process_rir(rir[0], config.sample_rate)

        generated_rir = fins.generate(_rir, config)

        output_filename = os.path.join(output_dir, filename) 

        assert generated_rir.shape == _rir.shape

        sf.write(output_filename, generated_rir, config.sample_rate)


if __name__ == "__main__" : 
    main()