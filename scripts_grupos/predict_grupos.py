import tensorflow.keras as keras
import librosa
import pyaudio
from scipy.signal import firwin, lfilter
from scipy.io.wavfile import write
import numpy as np
from pydub import AudioSegment

NUM_SAMPLES_TO_CONSIDER = 22050 # 1 segundo de áudio
MODEL_PATH = "../scripts_grupos/model.h5"

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
        "grupo1",
        "grupo2",
        "grupo3",
        "grupo4",
        "grupo5",
        "grupo6",
        "grupo7",
        "grupo8"
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


    def preprocess (self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):

        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        if len(signal) >= NUM_SAMPLES_TO_CONSIDER:

            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

            # filtro passa baixo
            lowpass_taps = 12
            cutoff_freq = 4000

            taps = firwin(lowpass_taps, cutoff_freq, window='hanning', pass_zero=True, fs=sr)
            signal = lfilter(taps, 1.0, signal)

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)


        else:
            audio_pad(file_path)

            signal_padded, sr = librosa.load('../buffers/padded_predict.wav')

            signal_padded = signal_padded[:NUM_SAMPLES_TO_CONSIDER]

            # filtro passa baixo
            lowpass_taps = 12
            cutoff_freq = 4000

            taps = firwin(lowpass_taps, cutoff_freq, window='hanning', pass_zero=True, fs=sr)
            signal_padded = lfilter(taps, 1.0, signal_padded)

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

    pad1_ms = 100  # milliseconds of silence needed
    pad2_ms = 1000
    silence1 = AudioSegment.silent(duration=pad1_ms)
    silence2 = AudioSegment.silent(duration=pad2_ms)
    audio = AudioSegment.from_wav(file_path)
    padded = silence1 + audio + silence2   # Adding silence after the audio

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

            write('../teste_grupos/palavra.wav', 44100, palavra)

            keyword = kss.predict("../teste_grupos/palavra.wav")

            print(f"{keyword}", "\n")

    #testar com ficheiros pré gravados de todos os grupos

    #keyword1 = kss.predict("../teste_grupos/G4.wav")
    #keyword2 = kss.predict("../teste_grupos/G2.wav")
    #keyword3 = kss.predict("../teste_grupos/G7.wav")
    #keyword4 = kss.predict("../teste_grupos/G3.wav")
    #keyword5 = kss.predict("../teste_grupos/G6.wav")
    #keyword6 = kss.predict("../teste_grupos/G8.wav")
    #keyword7 = kss.predict("../teste_grupos/G1.wav")
    #keyword8 = kss.predict("../teste_grupos/G5.wav")
    #print(f"Keywords: {keyword1}, {keyword2}, {keyword3}, {keyword4}, {keyword5}, {keyword6}, {keyword7}, {keyword8}")