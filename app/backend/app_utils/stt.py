import ffmpeg
import numpy as np
import whisper
from app_utils import load_stt_model


def load_audio_w_bytes(file: bytes, sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: (str, bytes)
        The audio file to open or bytes of audio file

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    if isinstance(file, bytes):
        inp = file
        file = "pipe:"
    else:
        inp = None

    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=inp)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def predict_stt(soundfile: bytes):
    model = load_stt_model()
    # model = whisper.load_model("base")
    audio = load_audio_w_bytes(soundfile)
    results = whisper.transcribe(model, audio)
    return results["segments"]


# f = open('../test2.mp3','rb')
# file = f.read()
# print(predict_stt(file))
