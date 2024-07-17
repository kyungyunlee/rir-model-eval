import os
import librosa 
import soundfile as sf 
import pandas as pd
from pathlib import Path
from models.model_utils import process_rir
import models.echo_density.echo_density as echo_density

FS = 48000
FILTER_FREQUENCIES = [250, 500, 1000, 2000, 4000, 8000]


def main() : 
    version_rir_df = pd.read_csv(f"./datasets/rir_list.csv", delimiter=",")

    for i, series in version_rir_df.iterrows() : 
        path, filename, rt, real  = series
        rir_file = Path("./") / path / filename

        rir, _ = librosa.load(rir_file, sr=FS)
        rir = process_rir(rir, FS)

        generated_rir = echo_density.generate(rir)
        
        assert generated_rir.shape == rir.shape

        output_dir = "models/echo_density/result"
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir, exist_ok=False)

        output_name = Path(output_dir) / filename
        sf.write(output_name, generated_rir, FS)

if __name__ == "__main__" : 
    main()