import librosa
import os
import json
import numpy as np
from pydub import AudioSegment
from scipy.fftpack import fft,fftfreq
from scipy.signal import firwin, lfilter, freqz,stft,istft
from scipy.io.wavfile import read




DATASET_PATH = "../dataset_comandos"
JSON_PATH = "../dataset_comandos/data.json"
SAMPLES_TO_CONSIDER = 22050 # 1 segundo de áudio (librosa importa áudio com 22050 Hz de sample rate)


def audio_pad(file_path):
    pad_ms = 1000  # milliseconds of silence needed
    silence = AudioSegment.silent(duration=pad_ms)
    audio = AudioSegment.from_wav(file_path)
    padded = audio + silence  # Adding silence after the audio

    padded.export('../buffers/padded_dataset.wav', format='wav')



if __name__ == "__main__":

    n_fft = 1030
    hop_length = 512

    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATASET_PATH)):

        #dirnames[:] = [d for d in dirnames if not d.startswith('test') if not d.startswith('buffers')]

        if dirpath is not DATASET_PATH:

            dirpath = dirpath.replace(os.sep, '/')
            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            for name in filenames:

                if name.endswith('.wav'):

                    file_path = os.path.join(dirpath, name)

                    fs, signal = read(file_path)

                    if len(signal) >= SAMPLES_TO_CONSIDER:

                        signal = signal[:SAMPLES_TO_CONSIDER]

                        # aplicar filtro de ênfase de altas frequências
                        #taps = 45
                        #Lo_freq = 2000
                        #High_freq = 7000
                        #signal = 4 * signal

                        #taps2 = firwin(taps, [Lo_freq, High_freq], window='hanning', pass_zero=False, fs=sr)  # high pass coeficients
                        #signal = lfilter(taps2, 1.0, signal)

                        f, t, Zxx = stft(signal, fs=fs, window='hanning', nperseg=n_fft, noverlap=hop_length)

                        Zxx = np.abs(Zxx.real)

                        MFCCs = np.zeros([13, len(t)], dtype=np.float32)  # iniciate an array to store the coeficients

                        for j in range(len(t)):
                            MFCCs[0, j] = np.sum(Zxx[0:8, j])  # 0...300
                            MFCCs[1, j] = np.sum(Zxx[8:13, j])  # 300 ... 517
                            MFCCs[2, j] = np.sum(Zxx[13:18, j])  # 517 ... 781
                            MFCCs[3, j] = np.sum(Zxx[18:26, j])  # 781 ... 1103
                            MFCCs[4, j] = np.sum(Zxx[26:35, j])  # 1003 ... 1496
                            MFCCs[5, j] = np.sum(Zxx[35:46, j])  # 1496 ... 1973
                            MFCCs[6, j] = np.sum(Zxx[46:59, j])  # 1973 ... 2254
                            MFCCs[7, j] = np.sum(Zxx[59:76, j])  # 2254 ... 3261
                            MFCCs[8, j] = np.sum(Zxx[76:96, j])  # 3261 ... 4122
                            MFCCs[9, j] = np.sum(Zxx[96:120, j])  # 4122 ... 5170
                            MFCCs[10, j] = np.sum(Zxx[120:150, j])  # 5170 ... 6446
                            MFCCs[11, j] = np.sum(Zxx[150:186, j])  # 6446 ... 8000
                            MFCCs[12, j] = np.sum(Zxx[186:233, j])  # 8000 ... 10021

                        #MFCCs = np.round(MFCCs, 3)

                        print(MFCCs.shape)

                        data["MFCCs"].append(MFCCs.T.tolist())
                        data["labels"].append(i - 1)
                        data["files"].append(file_path)
                        print("{}: {}".format(file_path, i - 1))

                    else:
                        audio_pad(file_path)

                        signal_padded, sr = librosa.load('../buffers/padded_dataset.wav')

                        signal_padded = signal_padded[:SAMPLES_TO_CONSIDER]

                        # aplicar filtro de ênfase de altas frequências
                        #taps = 45
                        #Lo_freq = 2000
                        #High_freq = 7000
                        #signal_padded = 4 * signal_padded

                        #taps2 = firwin(taps, [Lo_freq, High_freq], window='hanning', pass_zero=False, fs=sr)  # high pass coeficients
                        #signal_padded = lfilter(taps2, 1.0, signal_padded)

                        f, t, Zxx = stft(signal, fs=fs, window='hanning', nperseg=n_fft, noverlap=hop_length)

                        Zxx = np.abs(Zxx.real)

                        MFCCs = np.zeros([13, len(t)], dtype=np.float32)  # iniciate an array to store the coeficients

                        for j in range(len(t)):
                            MFCCs[0, j] = np.sum(Zxx[0:8, j])  # 0...300
                            MFCCs[1, j] = np.sum(Zxx[8:13, j])  # 300 ... 517
                            MFCCs[2, j] = np.sum(Zxx[13:18, j])  # 517 ... 781
                            MFCCs[3, j] = np.sum(Zxx[18:26, j])  # 781 ... 1103
                            MFCCs[4, j] = np.sum(Zxx[26:35, j])  # 1003 ... 1496
                            MFCCs[5, j] = np.sum(Zxx[35:46, j])  # 1496 ... 1973
                            MFCCs[6, j] = np.sum(Zxx[46:59, j])  # 1973 ... 2254
                            MFCCs[7, j] = np.sum(Zxx[59:76, j])  # 2254 ... 3261
                            MFCCs[8, j] = np.sum(Zxx[76:96, j])  # 3261 ... 4122
                            MFCCs[9, j] = np.sum(Zxx[96:120, j])  # 4122 ... 5170
                            MFCCs[10, j] = np.sum(Zxx[120:150, j])  # 5170 ... 6446
                            MFCCs[11, j] = np.sum(Zxx[150:186, j])  # 6446 ... 8000
                            MFCCs[12, j] = np.sum(Zxx[186:233, j])  # 8000 ... 10021

                        #MFCCs = np.round(MFCCs, 3)

                        print(MFCCs.shape)

                        data["MFCCs"].append(MFCCs.T.tolist())
                        data["labels"].append(i - 1)
                        data["files"].append(file_path)
                        print("{}: {}".format(file_path, i - 1))


    with open(JSON_PATH, "w") as fp:
        json.dump(data, fp, indent=4)