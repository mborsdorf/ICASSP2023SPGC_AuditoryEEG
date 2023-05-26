Multi-Head Attention and GRU for Improved Match-Mismatch Classification of Speech Stimulus and EEG Response
---
Our solution (2nd Place) for the ICASSP 2023 Signal Processing Grand Challenge - Auditory EEG Decoding Challenge (Task 1 - Match-Mismatch). In this repository, you can find all our training and evaluation scripts.

Our solution is based on the challenge's baseline model. For further details please check the original paper

> B. Accou, M. Jalilpour Monesi, J. Montoya, H. Van hamme and T. Francart, "Modeling the relationship between acoustic stimulus and EEG with a dilated convolutional neural network," 2020 28th European Signal Processing Conference (EUSIPCO), Amsterdam, Netherlands, 2021, pp. 1175-1179, doi: 10.23919/Eusipco47968.2020.9287417.

as well as the corresponding model from the challenge's [GitHub repository](https://github.com/exporl/auditory-eeg-challenge-2023-code).


## Data
The challenge's data was provided as preprocessed speech stimuli (audio) data and elicited EEG responses. It is based on the following dataset:

> Lies Bollens, Bernd Accou, Hugo Van hamme, and Tom
Francart, “A Large Auditory EEG decoding dataset,”
https://doi.org/10.48804/K3VSND, KU Leuven RDR, V1, 2023.

Further details on the data preprocessing (EEG data, speech envelopes, and mel-spectrograms) are given and referenced in our challenge paper (please see the bottom of this web page).

## Description of Files
In the following, all our scripts and their purposes are briefly described.

#### experiment_models.py
Definition of the different models, that can be used.

#### training.py
Script to run the training. (Un)comment sections in order to train a specific model and to select between speech envelope or mel-spectrogram. Please change the data_folder path to the path of the preprocessed "split_data" folder. You can also change the path for the results folder.

#### envelope.py
Can be used to calculate the speech envelope from the raw audio stimulus.

#### mel_spectrogram.py
Can be used to calculate the mel-spectrogram of the raw audio stimulus. For the calculation of the mel-spectrograms of the final test data (test set 1 and 2), the raw audio signals are padded with zeros so that the final mel-spectrogram has a duration of 192 samples (3 sec) to match the mel-spectrogram length of the training data.
> audio = np.pad(audio, (0, 1298), 'constant') # 1298 zeros exactly leads to a mel-spectrogram duration of 192 samples

#### speech_features.py
Calls the calculate_envelope and calculate_mel_spectrogram functions. Please set the paths to the source_stimuli_folder (the test data "stimuli_segments") and the output folder for the "envelopes" and "mel_segments" accordingly.
We use this script only to calculate the mel-spectrograms of the final test data (test set 1 and 2).

#### dataset_generator.py
Script to implement the dataset generator, used for training, validation, and testing. Please change the feature dimension to (64,1,1) when using speech envelopes and to (64,28,28) when using mel-spectrograms.

#### plot_test_results.py
Used to open an already calculated eval.json file and to print + save the results properly. Please add the path to the results folder.

#### predict_test.py
Script to predict the performance of a model on the final test set. (Un)comment sections in order to evaluate specific models and set the path to the model folder. (Un)comment section to change between speech envelopes or mel-spectrograms. Set the path to either the envelopes or the mel-spectrograms. Stores the results in the correct format.

#### icassp.yml
This .yml file can be used to install a conda environment for running the experiments.
Please change the path in the icassp.yml file according to your system.
> prefix: ADD_PATH_HERE/anaconda3/envs/icassp

## Our Paper
If you enjoyed working with our solution, please cite us:
```
@INPROCEEDINGS{10096959,
  author={Borsdorf, Marvin and Pahuja, Saurav and Ivucic, Gabriel and Cai, Siqi and Li, Haizhou and Schultz, Tanja},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Multi-Head Attention and GRU for Improved Match-Mismatch Classification of Speech Stimulus and EEG Response},
  year={2023},
  volume={},
  number={},
  pages={1-2},
  doi={10.1109/ICASSP49357.2023.10096959}}

```
