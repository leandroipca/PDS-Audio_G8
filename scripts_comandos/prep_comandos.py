import librosa
import os
import json
import numpy as np
from pydub import AudioSegment
from scipy.signal import firwin, lfilter


DATASET_PATH = "../dataset_comandos"
JSON_PATH = "../dataset_comandos/data.json"
SAMPLES_TO_CONSIDER = 22050 # 1 segundo de áudio (librosa importa áudio com 22050 Hz de sample rate)


def audio_pad(file_path):
    pad_ms = 1000  # milisegundos de silêncio
    silence = AudioSegment.silent(duration=pad_ms)
    audio = AudioSegment.from_wav(file_path)
    padded = audio + silence  # adicionar silêncio ao sinal

    padded.export('../buffers/padded_dataset.wav', format='wav')



if __name__ == "__main__":

    n_mfcc = 13
    n_fft = 1024 # janela de 46ms
    hop_length = 256 # overlap de 75%

    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATASET_PATH)):

        if dirpath is not DATASET_PATH:

            dirpath = dirpath.replace(os.sep, '/')
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            for name in filenames:

                if name.endswith('.wav'):

                    file_path = os.path.join(dirpath, name)

                    signal, sr = librosa.load(file_path)

                    if len(signal) >= SAMPLES_TO_CONSIDER:

                        signal = signal[:SAMPLES_TO_CONSIDER]

                        # aplicar filtro passa banda com ênfase em altas frequências
                        #taps = 45
                        #Lo_freq = 2000
                        #High_freq = 7000
                        #signal = 4 * signal

                        #taps2 = firwin(taps, [Lo_freq, High_freq], window='hanning', pass_zero=False, fs=sr)
                        #signal = lfilter(taps2, 1.0, signal)

                        MFCCs = librosa.feature.mfcc(signal, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

                        MFCCs = np.round(MFCCs, 3)

                        data["MFCCs"].append(MFCCs.T.tolist())
                        data["labels"].append(i - 1)
                        data["files"].append(file_path)
                        print("{}: {}".format(file_path, i - 1))

                    else:
                        audio_pad(file_path)

                        signal_padded, sr = librosa.load('../buffers/padded_dataset.wav')

                        signal_padded = signal_padded[:SAMPLES_TO_CONSIDER]

                        # aplicar filtro passa banda com ênfase em altas frequências
                        #taps = 45
                        #Lo_freq = 2000
                        #High_freq = 7000
                        #signal_padded = 4 * signal_padded

                        #taps2 = firwin(taps, [Lo_freq, High_freq], window='hanning', pass_zero=False, fs=sr)
                        #signal_padded = lfilter(taps2, 1.0, signal_padded)

                        # extract MFCCs
                        MFCCs = librosa.feature.mfcc(signal_padded, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

                        MFCCs = np.round(MFCCs, 3)

                        data["MFCCs"].append(MFCCs.T.tolist())
                        data["labels"].append(i - 1)
                        data["files"].append(file_path)
                        print("{}: {}".format(file_path, i - 1))


    with open(JSON_PATH, "w") as fp:
        json.dump(data, fp, indent=4)


