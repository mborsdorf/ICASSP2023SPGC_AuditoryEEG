"""Run preprocessing on the stimuli."""
import glob
import json
import logging
import os

import numpy as np

from envelope import calculate_envelope
from mel_spectrogram import calculate_mel_spectrogram

if __name__ == "__main__":

    # Whether to overwrite already existing features
    overwrite = False

    # Run the extraction
    logging.info(f'Extracting speech features')

    # Get the path to the raw speech stimulus segments
    source_stimuli_folder = 'PATH_TO_THE_RAW_SPEECH_STIMULUS_SEGMENTS/stimuli_segments'
    # Get the path to save the preprocessed files
    output_stimuli_folder = 'PATH_TO_SAVE_THE_MEL_SEGMENTS_FOR_FINAL_EVALUATION/mel_segments'

    # Create the save directory if it didn't exist already
    os.makedirs(output_stimuli_folder, exist_ok=True)

    # Find the stimuli files
    speech_files = glob.glob(os.path.join(source_stimuli_folder, "*.npz"))

    # Preprocess the stimuli
    nb_speech_files = len(speech_files)
    logging.info("Found %u stimuli files", nb_speech_files)
    for index, filepath in enumerate(speech_files):
        # Loop over each speech file and create envelope and mel spectrogram
        # and save them
        filename = os.path.basename(filepath)

        print(f"Preprocessing {filepath} ({index + 1}/{nb_speech_files})")

        # # Envelope
        # env_path = os.path.join(output_stimuli_folder, filename.replace(".npz", "_envelope.npy"))
        # if not os.path.exists(env_path) or overwrite:
        #     envelope = calculate_envelope(filepath)
        #     np.save(env_path, envelope)
        # else:
        #     print(f"Skipping {env_path} because it already exists")

        # Mel
        mel_path = os.path.join(output_stimuli_folder, filename.replace(".npz", "_mel.npy"))
        if not os.path.exists(mel_path) or overwrite:
            mel = calculate_mel_spectrogram(filepath)
            np.save(mel_path, mel)
        else:
            print(f"Skipping {mel_path} because it already exists")

