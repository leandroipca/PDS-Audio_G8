import tensorflow.keras as keras
import librosa
import pyaudio
from scipy.signal import firwin, lfilter
from scipy.io.wavfile import write
import numpy as np
from pydub import AudioSegment

NUM_SAMPLES_TO_CONSIDER = 22050 # 1 segundo de áudio
MODEL_PATH = "../scripts_comandos/model.h5"

CHUNK = 1024 *4 # lenght of input buffer fast and without cuts
FORMAT = pyaudio.paInt16  # format of .wav file could be paInt16/paFloat32
CHANNELS = 1 # mono
RATE = 44100 # sampling rate

p = pyaudio.PyAudio() # create a pyaudio object
stream = p.open(format = FORMAT,
               channels = CHANNELS,
                rate = RATE,
                input = True,
                output = True,
                frames_per_buffer = CHUNK)

recording = False  # flag to enable sound recording
record_seconds = 1
palavra = np.array([],dtype=np.int16)   # np dinamic array



class _Keyword_Spotting_Service:

    model = None
    mappings = [
        "avancar",
        "baixo",
        "centro",
        "cima",
        "direita",
        "esquerda",
        "parar",
        "recuar"
    ]
    _instance = None

    def predict(self, file_path):

        # extract the MFCCs
        MFCCs = self.preprocess(file_path) # (# segments, # coefficients)

        # convert 2d MFCCs array to 4d array -> (# samples, # segments, # coefficients, # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(MFCCs) # [ [0.1, 0.6, 0.1, (...)] ]
        predicted_index = np.argmax(predictions)
        predicted_keyword = self.mappings[predicted_index]

        return predicted_keyword


    def preprocess (self, file_path, n_mfcc=13, n_fft=1024, hop_length=256):

        # load audio file
        signal, sr = librosa.load(file_path)

        if len(signal) >= NUM_SAMPLES_TO_CONSIDER:

            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

            # aplicar filtro passa banda com ênfase em altas frequências
            #taps = 45
            #Lo_freq = 2000
            #High_freq = 7000
            #signal = 4 * signal

            #taps2 = firwin(taps,[Lo_freq,High_freq],window='hanning',pass_zero=False,fs=sr)  # high pass coeficients
            #signal = lfilter(taps2, 1.0, signal)

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)


        else:
            audio_pad(file_path)

            signal_padded, sr = librosa.load('../buffers/padded_predict.wav')

            signal_padded = signal_padded[:NUM_SAMPLES_TO_CONSIDER]

            # aplicar filtro passa banda com ênfase em altas frequências
            #taps = 45
            #Lo_freq = 2000
            #High_freq = 7000
            #signal_padded = 4 * signal_padded

            #taps2 = firwin(taps, [Lo_freq, High_freq], window='hanning', pass_zero=False, fs=sr)  # high pass coeficients
            #signal_padded = lfilter(taps2, 1.0, signal_padded)

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal_padded, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)


        return MFCCs.T


def Keyword_Spotting_Service():

    # ensure thar we only have one instance of key spotting service
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance


def audio_pad(file_path):

    pad1_ms = 100  # milisegundos de áudio a adicionar
    pad2_ms = 1000
    silence1 = AudioSegment.silent(duration=pad1_ms)
    silence2 = AudioSegment.silent(duration=pad2_ms)
    audio = AudioSegment.from_wav(file_path)
    padded = silence1 + audio + silence2   # adicionar silêncio ao sinal

    padded.export('../buffers/padded_predict.wav', format='wav')


if __name__ == "__main__":

    kss = Keyword_Spotting_Service()

    while True:
        buffer = stream.read(CHUNK*3)  # pre_trigger audio
        buff = np.frombuffer(buffer, dtype=np.int16)  # converte em tempo real para numpy array
        data = stream.read(CHUNK)  # trigger and record audio
        data_int = np.frombuffer(data, dtype=np.int16)  # converte em tempo real para numpy array

        if (np.abs(data_int.max()) > 1000):  # caso encontre um valor absoluto superior 1000 inicia a gravacao
            #print(np.abs(data_int.max()))
            recording = True
            i = 0
            print("A gravar")

        if (recording == True):
            palavra = np.array([], dtype=np.int16)  # Restart array
            palavra = np.append(palavra, buff)  # Grava o primeiro valor pre_trigger
            palavra = np.append(palavra, data_int)  # adiciona os pedaco existente ao vetor

            for i in range(int(44100 / CHUNK * record_seconds)):  # grava os seguintes
                data = stream.read(CHUNK)
                data_int = np.frombuffer(data, dtype=np.int16)  # converte em tempo re
                palavra = np.append(palavra, data_int)  # adiciona a os pedacos ao vetor

            #print(len(palavra))
            recording = False
            print("Gravado\n")

            write('../teste_comandos/palavra.wav', 44100, palavra)

            keyword = kss.predict("../teste_comandos/palavra.wav")

            print(f"{keyword}", "\n")


