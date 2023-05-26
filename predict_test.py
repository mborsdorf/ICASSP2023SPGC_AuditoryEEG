"""
Sample code to generate labels for test dataset of
match-mismatch task. The requested format for submitting the labels is
as follows:
for each subject a json file containing a python dictionary in the
format of  ==> {'sample_id': prediction, ... }.
"""

import os
import glob
import json
import numpy as np
import envelope
#from experiment_models import dilation_model                 # Baseline model
#from experiment_models import eeg_mha_dc_speech_dc_model     # MHA+DC for EEG and DC for speech stimulus
from experiment_models import eeg_mha_dc_speech_gru_dc_model # MHA+DC for EEG and GRU+DC for speech stimulus


def create_test_samples(eeg_path, envelope_dir):
    with open(eeg_path, 'r') as f:
        sub = json.load(f)
    eeg_data = []
    spch1_data = []
    spch2_data = []
    id_list = []
    for key, sample in sub.items():
        eeg_data.append(sample[0])

        spch1_path = os.path.join(envelope_dir, sample[1])
        spch2_path = os.path.join(envelope_dir, sample[2])

        # For using envelope
        # envelope1 = np.load(spch1_path)
        # env1 = envelope1['envelope']
        # envelope2 = np.load(spch2_path)
        # env2 = envelope2['envelope']

        # For using calculated mel-spectrograms
        # Load mel spectrograms
        spch1_path = spch1_path.replace(".npz", "_mel.npy")
        env1 = np.load(spch1_path, allow_pickle=True)
        spch2_path = spch2_path.replace(".npz", "_mel.npy")
        env2 = np.load(spch2_path, allow_pickle=True)

        spch1_data.append(env1)
        spch2_data.append(env2)
        id_list.append(key)
    eeg = np.array(eeg_data)
    spch1 = np.array(spch1_data)
    spch2 = np.array(spch2_data)
    return (eeg, spch1, spch2), id_list


def get_label(pred):
    if pred >= 0.5:
        label = 1
    else:
        label = 0
    return label

if __name__ == '__main__':

    window_length = 3*64
    stimulus_dimension = 1 # Envelope=1 ; Mel-spectrogram=28

    # Root dataset directory containing test set
    # Change the path to the downloaded test dataset dir
    dataset_dir = 'PATH_TO_EVALUATION_DATA'

    # Path to your pretrained model
    pretrained_model = os.path.join('PATH_PRETRAINED_MODEL', 'model.h5')

    # # Calculate envelope of the speech files (only if the envelope directory does not exist)
    # stimuli_dir = os.path.join(dataset_dir, 'stimuli_segments')
    # envelope_dir = os.path.join(dataset_dir, 'envelope_segments')
    # # Create envelope of segments if it has not already been created
    # if not os.path.isdir(envelope_dir):
    #     os.makedirs(envelope_dir, exist_ok=True)
    # for stimulus_seg in glob.glob(os.path.join(stimuli_dir, '*.npz')):
    #     base_name = os.path.basename(stimulus_seg).split('.')[0]
    #     if not os.path.exists(os.path.join(envelope_dir, base_name + '.npz')):
    #         env = envelope.calculate_envelope(stimulus_seg)
    #         target_path = os.path.join(envelope_dir, base_name + '.npz')
    #         np.savez(target_path, envelope=env)


    # Define and load the pretrained model
    #model = dilation_model(time_window=window_length, eeg_input_dimension=64, env_input_dimension=stimulus_dimension)                 # Baseline model
    #model = eeg_mha_dc_speech_dc_model(time_window=window_length, eeg_input_dimension=64, env_input_dimension=stimulus_dimension)     # MHA+DC for EEG and DC for speech stimulus
    model = eeg_mha_dc_speech_gru_dc_model(time_window=window_length, eeg_input_dimension=64, env_input_dimension=stimulus_dimension) # MHA+DC for EEG and GRU+DC for speech stimulus
    model.load_weights(pretrained_model)

    test_data = glob.glob(os.path.join(dataset_dir, 'sub*.json'))
    for sub_path in test_data:
        subject = os.path.basename(sub_path).split('.')[0]

        sub_dataset, id_list = create_test_samples(sub_path, os.path.join(dataset_dir, 'FOLDER_FOR_CALCULATED_MEL-SPECTROGRAMS_OR_ENVELOPES')) # For using mel-spectrograms use the /mel_segments folder from "speech_features.py"
        # Normalize data
        subject_data = []
        for item in sub_dataset:
            item_mean = np.expand_dims(np.mean(item, axis=1), axis=1)
            item_std = np.expand_dims(np.std(item, axis=1), axis=1)
            subject_data.append((item - item_mean) / item_std)
        sub_dataset = tuple(subject_data)

        predictions = model.predict(sub_dataset)
        predictions = list(np.squeeze(predictions))
        predictions = map(get_label, predictions)
        sub = dict(zip(id_list, predictions))

        prediction_dir = os.path.join(os.path.dirname(__file__), 'predictions')
        os.makedirs(prediction_dir, exist_ok=True)
        with open(os.path.join(prediction_dir, subject + '.json'), 'w') as f:
            json.dump(sub, f)