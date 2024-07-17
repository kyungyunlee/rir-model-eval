import os
import soundfile as sf
import librosa
import pandas as pd 
from pathlib import Path
from models.model_utils import process_rir
from models.fdn import fdn 



def main():

    fs = 48000 
    fBands = [1, 63, 125, 250, 500, 1000, 2000, 4000, 8000, fs]
    n_slopes = 1

    version_rir_df = pd.read_csv(f"./datasets/rir_list.csv", delimiter=",")

    for i, series in version_rir_df.iterrows() : 
        path, filename, rt, real  = series
        print (path, filename, rt, real)
        rir_file = Path(path) / filename 
         
        # Load RIR 
        rir, _ = librosa.load(rir_file, sr=fs)
        
        if len(rir.shape) > 1:
            rir = rir[:, 0]

        rir = process_rir(rir, fs)

        generated_rir = fdn.generate(rir, fBands, fs)
       
        assert generated_rir.shape == rir.shape

        # Save
        output_dir = "models/fdn/result"
        if not os.path.exists(output_dir) : 
            os.makedirs(output_dir, exist_ok=False)
            
        sf.write(os.path.join(output_dir, filename), generated_rir, fs)


if __name__ == '__main__':
    main()
