import os
import librosa
import pandas as pd
from pathlib import Path
import soundfile as sf
from models.model_utils import process_rir
import models.rt2rir.rt2rir as rt2rir


def main():
    version_rir_df = pd.read_csv(f"./datasets/rir_list.csv", delimiter=",")

    for i, series in version_rir_df.iterrows():
        path, filename, rt, real = series
        rir_file = Path("./") / path / filename

        rir, sr = librosa.load(rir_file, sr=48000)

        if len(rir.shape) > 1:
            rir = rir[0, :]

        rir = process_rir(rir, sr)

        filter_frequencies = [250, 500, 1000, 2000, 4000, 8000]

        full_generated_rir = rt2rir.generate(rir, sr, filter_frequencies)

        assert full_generated_rir.shape == rir.shape

        # Save
        output_dir = "models/rt2rir/result"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=False)

        sf.write(os.path.join(output_dir, filename), full_generated_rir, sr)


if __name__ == "__main__":
    main()