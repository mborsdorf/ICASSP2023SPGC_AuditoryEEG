"""Code to calculate mel spectrograms."""
import math

import librosa
import numpy as np
import scipy.signal


def calculate_mel_spectrogram(
    audio_path,
    target_fs=64,
    fmin=-4.2735,
    fmax=5444,
    nb_filters=28,
    hop_length=None,
    win_length=None,
):
    """Calculates mel spectrogram of a raw speech file.
    Parameters
    ---------
    audio_path: str
        audio file path
    target_fs: int
        Sampling frequency of the calculated mel spectrogram
    fmin: Union[float, int]
        Minimum center frequency used in mel filter matrix
    fmax: Union[float, int]
        Maximum center frequency used in mel filter matrix
    nb_filters: int
        Number of mel spectrogram frequency bands
    hop_length: int
        Hop length (in samples) used for calculation of the spectrogram
    win_length: int
        Window length (in samples) of each frame
    Returns
    -------
    numpy.ndarray
        Mel spectrogram
    """

    speech = np.load(audio_path)
    audio, fs = speech["audio"], speech["fs"]
    print(audio.shape)
    print(fs)
    if not hop_length:
        hop_length = int((1 / target_fs) * fs)  # this will downsample the signal to target_fs Hz
    if not win_length:
        win_length = int(0.025 * fs)  # 25 milli seconds

    print(hop_length)
    print(win_length)

    # Finds the closest power of 2
    # that is bigger than win_length
    n_fft = int(math.pow(2, math.ceil(math.log2(win_length))))
    print(n_fft)
    # DC removal
    audio = audio - np.mean(audio)
    
    
    # For the calculation of the mel-spectrograms of the final test data (test set 1 and 2) the raw audio signals are padded with zeros
    # so that the final mel-spectrogram has a duration of 192 samples (3 sec) to match the mel-spectrogram length of the training data
    
    # audio = np.pad(audio, (0, 1298), 'constant') # 1298 zeros exactly leads to a mel-spectrogram duration of 192 samples

    mel_spectrogram = librosa.feature.melspectrogram(audio, window=scipy.signal.windows.hamming(win_length),
                                       sr=fs, n_fft=n_fft, hop_length=hop_length,
                                       win_length=win_length, fmin=fmin, fmax=fmax, htk=True,
                                       n_mels=nb_filters, center=False, norm=None, power=1.0).T

    print(mel_spectrogram.shape)
    mel_spectrogram=np.power(mel_spectrogram, 0.6)

    return mel_spectrogram