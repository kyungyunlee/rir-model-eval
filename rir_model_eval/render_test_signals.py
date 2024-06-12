import os
from pathlib import Path
import argparse
import pandas as pd
import random
import ruamel.yaml
from scipy.signal import fftconvolve
import soundfile as sf
import numpy as np
import shutil
from utils.audio import load_audio
from models.model_utils import process_rir

TARGET_DB = -30
SPEECH_FILE_DIR = "./datasets/speech/audio_exports_Harvard_Sets_Cleaned_Extended"
SPEAKER_LIST = np.array([f"P{p}" for p in range(1, 22)])
FS = 48000


def rms(sig):
    return np.sqrt(np.mean(sig**2))


def loudness_normalize(sig, target_db):
    curr_db = 20 * np.log10(rms(sig))
    diff_db = target_db - curr_db
    gain_factor = 10 ** (diff_db / 20)
    # new_db = 20 * np.log10(rms(synth_rir_convolved*gain_factor))
    return sig * gain_factor


def is_clipping(sig):
    if np.max(np.abs(sig)) > 1:
        return True
    return False


def fix_audio_length(sig, target_length):
    # Sig shape = (1, length)
    if sig.shape[1] < target_length:
        out = np.zeros((1, target_length))
        out[:, : sig.shape[1]] = sig
    elif sig.shape[1] > target_length:
        out = sig[:, :target_length]
    else:
        out = sig
    return out


def remove_silence_at_end(sig, fs):
    win_len = int(fs * 0.002)
    pad_extra = int(fs * 0.01)

    overlap = 0.75
    energy_threshold = 1e-6
    win = np.hanning(win_len)

    is_2channel = False
    if len(sig.shape) > 1:
        # 2 channel
        sig = sig[0]
        is_2channel = True

    sig = np.pad(sig, (int(win_len * overlap), int(win_len * overlap)))
    hop = 1 - overlap

    n_wins = np.floor(sig.shape[0] / (win_len * hop) - 1 / 2 / hop)

    local_energy = []
    for i in range(1, int(n_wins - 1)):
        local_energy.append(
            np.sum(
                (
                    sig[
                        (i - 1) * int(win_len * hop) : (i - 1) * int(win_len * hop)
                        + win_len
                    ]
                    ** 2
                )
                * win
            )
        )

    # discard trailing points
    # remove (1/2/hop) to avoid map to negative time (center of window)
    n_win_discard = (overlap / hop) - (1 / 2 / hop)

    local_energy = np.array(local_energy[int(n_win_discard) :])

    loc = np.argwhere(local_energy > energy_threshold)
    end_location_in_samples = int(win_len * hop * loc[-1][0])

    if is_2channel:
        sig = sig[np.newaxis, : end_location_in_samples + pad_extra]
    else:
        sig = sig[: end_location_in_samples + pad_extra]

    return sig


def render(args):
    testing_models = args.models
    print(f"List of models for testing: {testing_models}")

    # Check models
    for model in testing_models:
        model_dir = os.path.join("models", model, "result")
        if not os.path.exists(model_dir):
            print(
                f"Wrong model name '{model}' or model directory does not exist: {model_dir}"
            )
            return

        # Also check if all the files are there

    # Add the anchor : Speech with no reverb
    # testing_models.append("anchor")

    # Load RIR eq
    rir_eq = load_audio("./datasets/rir_eq.wav", target_sr=FS, mono=True)

    counter = 0

    # Make output directory to save result
    output_dir = Path(args.output_dir) / args.output_filename
    if os.path.exists(output_dir):
        print(f"Directory {output_dir} already exists, removing...")
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Load RIR data
    rir_df = pd.read_csv(f"./datasets/rir_list.csv", delimiter=",")

    listening_test_data = {}

    for model in testing_models:
        print(f"Rendering for model {model}...")

        listening_test_data[model] = {}

        for n in range(len(rir_df)):
            listening_test_data[model][n] = {}

            # Current RIR
            rir_path, rir_file, rt60, real = rir_df.iloc[n]
            original_rir_file = Path(rir_path) / rir_file

            ####################### Choose speech for each type #######################
            chosen_speech_sets = {}

            ##### sameSpk-sameSent
            # Choose speaker
            curr_speaker = np.random.choice(SPEAKER_LIST)
            # Get all files for the speaker
            speaker_files = list(Path(SPEECH_FILE_DIR).glob(f"{curr_speaker}*.wav"))
            # Just select one file from the list
            chosen_file = random.choice(speaker_files).as_posix()
            chosen_speech_sets["sameSpk-sameSent"] = [
                chosen_file,
                chosen_file,
                chosen_file,
            ]

            ##### sameSpk-diffSent
            # # Choose speaker
            # curr_speaker = random.choice(SPEAKER_LIST)
            # # Get all files for the speaker
            # speaker_files = [f.as_posix() for f in list(Path(SPEECH_FILE_DIR).glob(f"{curr_speaker}*.wav"))]
            # three_random_indices = np.arange(0,len(speaker_files))
            # np.random.shuffle(three_random_indices)
            # three_random_indices = three_random_indices[:3]
            # chosen_files = np.array(speaker_files)[three_random_indices]
            # chosen_speech_sets["sameSpk-diffSent"] = chosen_files

            ##### diffSpk-diffSent
            # Choose speaker
            np.random.shuffle(SPEAKER_LIST)
            three_random_speakers = SPEAKER_LIST[:3]

            # Get all files for the speaker
            speaker1_files = [
                f.as_posix()
                for f in list(
                    Path(SPEECH_FILE_DIR).glob(f"{three_random_speakers[0]}*.wav")
                )
            ]
            speaker1_chosen_file = np.random.choice(speaker1_files)
            speaker1_set_sentence = " - ".join(
                speaker1_chosen_file.split(".wav")[0].split(" - ")[1:]
            )

            speaker2_files = [
                f.as_posix()
                for f in list(
                    Path(SPEECH_FILE_DIR).glob(f"{three_random_speakers[1]}*.wav")
                )
            ]
            speaker2_files = [
                f for f in speaker2_files if speaker1_set_sentence not in f
            ]
            speaker2_chosen_file = np.random.choice(speaker2_files)
            speaker2_set_sentence = " - ".join(
                speaker2_chosen_file.split(".wav")[0].split(" - ")[1:]
            )

            speaker3_files = [
                f.as_posix()
                for f in list(
                    Path(SPEECH_FILE_DIR).glob(f"{three_random_speakers[2]}*.wav")
                )
            ]
            speaker3_files = [
                f
                for f in speaker3_files
                if speaker1_set_sentence not in f and speaker2_set_sentence not in f
            ]
            speaker3_chosen_file = np.random.choice(speaker3_files)
            speaker3_set_sentence = " - ".join(
                speaker3_chosen_file.split(".wav")[0].split(" - ")[1:]
            )

            assert speaker1_set_sentence != speaker2_set_sentence
            assert speaker1_set_sentence != speaker3_set_sentence
            assert speaker2_set_sentence != speaker3_set_sentence

            chosen_speech_sets["diffSpk-diffSent"] = [
                speaker1_chosen_file,
                speaker2_chosen_file,
                speaker3_chosen_file,
            ]

            for source_type, (
                speaker1_file,
                speaker2_file,
                speaker3_file,
            ) in chosen_speech_sets.items():

                listening_test_data[model][n][source_type] = {}

                speaker_file_list = [speaker1_file, speaker2_file, speaker3_file]
                synthesized_rir_file = Path("models") / model / "result" / rir_file

                if model != "anchor" and not os.path.exists(synthesized_rir_file):
                    print(f"File does not exists: {synthesized_rir_file}")
                    return

                # Render
                synth_idx = random.randint(0, 2)

                # Load speech
                speech = load_audio(
                    speaker_file_list[synth_idx], target_sr=FS, mono=False
                )
                speech = speech[0:1, int(FS * 0.1) :]

                if model == "anchor":
                    synth_rir_equalized = speech

                else:
                    synthesized_rir = load_audio(
                        synthesized_rir_file, target_sr=FS, mono=True
                    )
                    original_rir = load_audio(
                        original_rir_file, target_sr=FS, mono=True
                    )
                    original_rir = process_rir(original_rir[0], FS)
                    original_rir = original_rir[np.newaxis, :]

                    if synthesized_rir.shape != original_rir.shape:

                        if synthesized_rir.shape[1] > original_rir.shape[1]:
                            synthesized_rir = synthesized_rir[
                                :, : original_rir.shape[1]
                            ]
                        else:
                            synthesized_rir_tmp = np.zeros_like(original_rir)
                            synthesized_rir_tmp[:, : synthesized_rir.shape[1]] = (
                                synthesized_rir
                            )
                            synthesized_rir = synthesized_rir_tmp

                    synth_rir_convolved = fftconvolve(
                        speech, synthesized_rir, mode="full"
                    )
                    synth_rir_equalized = fftconvolve(
                        synth_rir_convolved, rir_eq, mode="full"
                    )

                    # Remove silence at end
                    synth_rir_equalized = remove_silence_at_end(synth_rir_equalized, FS)

                # LOUDNESS NORMALIZE
                synth_rir_equalized = loudness_normalize(synth_rir_equalized, TARGET_DB)
                assert not is_clipping(synth_rir_equalized)

                synth_output_name = os.path.join(
                    output_dir,
                    f"Q{counter+1}__{Path(rir_path).stem}__{Path(rir_file).stem}__{source_type}__{Path(speaker_file_list[synth_idx]).stem.replace(' ', '_')}__{model}.wav",
                )

                # Render other 2 speakers with original RIR
                original_idxs = [0, 1, 2]
                original_idxs.remove(synth_idx)

                original_rir_equalized_list = []
                original_output_name_list = []

                for k, idx in enumerate(original_idxs):
                    speech = load_audio(
                        speaker_file_list[idx], target_sr=FS, mono=False
                    )
                    speech = speech[0:1, int(FS * 0.1) :]

                    original_rir_convolved = fftconvolve(
                        speech, original_rir, mode="full"
                    )
                    original_rir_equalized = fftconvolve(
                        original_rir_convolved, rir_eq, mode="full"
                    )
                    original_rir_equalized = remove_silence_at_end(
                        original_rir_equalized, FS
                    )

                    # SAVE
                    original_output_name = os.path.join(
                        output_dir,
                        f"Q{counter+1}__{Path(rir_path).stem}__{Path(rir_file).stem}__{source_type}__{Path(speaker_file_list[idx]).stem.replace(' ', '_')}__original{k+1}.wav",
                    )
                    listening_test_data[model][n][source_type][
                        f"original{k+1}"
                    ] = original_output_name
                    # LOUDNESS NORMALIZE
                    original_rir_equalized = loudness_normalize(
                        original_rir_equalized, TARGET_DB
                    )

                    assert not is_clipping(original_rir_equalized)

                    original_output_name_list.append(original_output_name)
                    original_rir_equalized_list.append(original_rir_equalized)

                listening_test_data[model][n][source_type][
                    "generated"
                ] = synth_output_name

                counter += 1

                # Save audio
                # get the longest audio file.
                max_audio_length = max(
                    synth_rir_equalized.shape[1],
                    original_rir_equalized_list[0].shape[1],
                    original_rir_equalized_list[1].shape[1],
                )

                synth_rir_equalized = fix_audio_length(
                    synth_rir_equalized, max_audio_length
                )
                original_rir_equalized_list[0] = fix_audio_length(
                    original_rir_equalized_list[0], max_audio_length
                )
                original_rir_equalized_list[1] = fix_audio_length(
                    original_rir_equalized_list[1], max_audio_length
                )

                assert (
                    synth_rir_equalized.shape[1]
                    == original_rir_equalized_list[0].shape[1]
                    == original_rir_equalized_list[1].shape[1]
                )

                # make into 2 channels
                synth_rir_equalized_2ch = np.vstack(
                    [synth_rir_equalized, synth_rir_equalized]
                )
                sf.write(synth_output_name, synth_rir_equalized_2ch.T, FS)

                for i in range(2):
                    original_rir_equalized_2ch = np.vstack(
                        [original_rir_equalized_list[i], original_rir_equalized_list[i]]
                    )
                    sf.write(
                        original_output_name_list[i], original_rir_equalized_2ch.T, FS
                    )

    return listening_test_data


def main(args):

    listening_test_data = render(args)

    # Save in mushra format
    question_bank = []
    for model, questions_per_type in listening_test_data.items():
        for question_type, question in questions_per_type.items():
            for i, qdata in question.items():
                question_bank.append(qdata)

    random.shuffle(question_bank)

    # Make a dictionary for yaml
    # Add basic info for Mushra test
    mushra_dict = {}
    mushra_dict["testname"] = "RIR evaluation"
    mushra_dict["testId"] = args.output_filename
    mushra_dict["bufferSize"] = 2048
    mushra_dict["stopOnErrors"] = True
    mushra_dict["showButtonPreviousPage"] = True
    mushra_dict["remoteService"] = "service/write.php"

    # Add each set of questions as a Mushra "page"
    mushra_dict["pages"] = []
    for q in question_bank:
        page_dict = {}
        page_dict["type"] = "3AFC"
        page_dict["id"] = q["generated"].split("/")[-1]
        page_dict["name"] = "Generative RIR evaluation"
        # page_dict["unforced"] = ""
        page_dict["content"] = (
            "Which one is in a different room? Choose a random one if they all sound similar."
        )
        page_dict["showWaveform"] = False
        page_dict["enableLooping"] = True
        page_dict["stimuli"] = q
        mushra_dict["pages"].append(page_dict)

    # Add the finish page
    finish_dict = {}
    finish_dict["type"] = "finish"
    finish_dict["name"] = "Thank you"
    finish_dict["content"] = "Thank you for participating"
    finish_dict["showResults"] = False
    finish_dict["writeResults"] = True
    finish_dict["questionnaire"] = [
        {
            "type": "number",
            "label": "Participant ID",
            "name": "participantId",
            "min": 0,
            "max": 1000,
            "default": 0,
        },
        {
            "type": "number",
            "label": "Age",
            "name": "age",
            "min": 0,
            "max": 100,
            "default": 0,
        },
        {
            "type": "number",
            "label": "How many years have you spent doing academic research in acoustics?",
            "name": "experience",
            "min": 0,
            "max": 100,
            "default": 0,
        },
    ]
    mushra_dict["pages"].append(finish_dict)

    yaml = ruamel.yaml.YAML()
    yaml.indent(sequence=6, offset=4)

    with open(os.path.join(args.output_dir, f"{args.output_filename}.yaml"), "w") as f:
        d = yaml.dump(mushra_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", type=str, nargs="+", required=True, help="e.g. --model RT2RIR"
    )
    parser.add_argument(
        "--output_filename", type=str, required=True, help="Filename of the config file"
    )
    parser.add_argument("-O", "--output_dir", type=str, default="output", help="")

    args = parser.parse_args()

    main(args)
