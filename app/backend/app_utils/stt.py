from typing import List

import ffmpeg
import kss
import numpy as np
import whisper
from app_utils import load_small_stt_model
from hanspell import spell_checker


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
        # sudo apt install ffmpeg
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=inp)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def predict_stt(soundfile: bytes) -> list:
    model = load_small_stt_model()
    audio = load_audio_w_bytes(soundfile)
    stt_results = whisper.transcribe(model, audio)
    sentences = [seg["text"].strip() for seg in stt_results["segments"]]
    results = get_post_process(sentences)
    return results


def split_sentences(text: List[str]) -> List[str]:
    """
    500자 이상인 경우 hanspell.spell_checker() 함수가 동작하지 않으므로 500자 이하로 구분
    """
    content_list = [""]

    for t in text:
        if len(content_list[-1]) + len(t) < 500:
            content_list[-1] += t
        else:
            content_list.append(t)

    return content_list


def get_post_process(texts: List[str]):
    sents = split_sentences(texts)
    spelled_sents = spell_checker.check(sents)
    spelled_after = [spelled.checked for spelled in spelled_sents]

    result = []
    for i in spelled_after:
        result.extend(kss.split_sentences(i))  # 문장 분리
    result = [f"{r} " for r in result]  # 문장 간 띄어쓰기 추가
    return result
