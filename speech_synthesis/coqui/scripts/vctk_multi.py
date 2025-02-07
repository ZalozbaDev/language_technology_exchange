""" Created 2023

    @author: korla

Copyright (c) 2023 Korla Baier

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import random
import string
from alive_progress import alive_bar
import random


def get_random_string(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


DATA_DIRS = [
    {
        "path": "/home/lucija/coqui_tts_training/corpora/export/speech_corpus_michal_recordings/LJSpeech-1.1",
        "type": "ljspeech",
        "full_filename": False,
        "remove_silence": True,
    },
    {
        "path": "/home/lucija/coqui_tts_training/corpora/speech_corpus_weronika_recordings",
        "type": "ljspeech",
        "full_filename": False,
        "remove_silence": True,
    },
    {
        "path": "/home/lucija/coqui_tts_training/corpora/speech_corpus_korla_recordings/dataset",
        "type": "ljspeech",
        "full_filename": False,
        "remove_silence": True,
    },
    {
        "path": "/home/lucija/coqui_tts_training/corpora/speech_corpus_mic_recordings_1",
        "type": "zalozba_film",
        "remove_silence": False,
    },
    {
        "path": "/home/lucija/coqui_tts_training/corpora/speech_corpus_mic_recordings_2",
        "type": "zalozba_film",
        "remove_silence": False,
    },
    {
        "path": "/home/lucija/coqui_tts_training/corpora/speech_corpus_mic_recordings_3",
        "type": "zalozba_film",
        "remove_silence": False,
    },
    {
        "path": "/home/lucija/coqui_tts_training/corpora/speech_corpus_mic_recordings_4",
        "type": "zalozba_film",
        "remove_silence": False,
    },
    {
        "path": "/home/lucija/coqui_tts_training/corpora/speech_corpus_mic_recordings_5",
        "type": "zalozba_film",
        "remove_silence": False,
    },
    {
        "path": "/home/lucija/coqui_tts_training/corpora/speech_corpus_mic_recordings_6",
        "type": "zalozba_film",
        "remove_silence": False,
    },
    {
        "path": "/home/lucija/coqui_tts_training/corpora/speech_corpus_mic_recordings_7",
        "type": "zalozba_film",
        "remove_silence": False,
    },
    {
        "path": "/home/lucija/coqui_tts_training/corpora/speech_corpus_mic_recordings_8",
        "type": "zalozba_film",
        "remove_silence": False,
    },
    {
        "path": "/home/lucija/coqui_tts_training/corpora/speech_corpus_XXX",
        "type": "zalozba_film",
        "remove_silence": False,
    },
    {
        "path": "/home/lucija/coqui_tts_training/corpora/speech_corpus_YYY",
        "type": "zalozba_film",
        "remove_silence": False,
    },
    {
        "path": "/home/lucija/coqui_tts_training/corpora/speech_corpus_ZZZ",
        "type": "zalozba_film",
        "remove_silence": False,
    },
]

EXPORT_TO = "/home/lucija/coqui_tts_training/vctk_export"

# audio options
SAMPLE_RATE = 22050
AUDIO_FORMAT = "wav"

LOWERCASE = False

SILENCE_THRESHOLD = "-50d"


def procces_text(text: str):
    if LOWERCASE:
        text = text.lower()

    if not (text[-1] == "." or text[-1] == "!" or text[-1] == "?"):
        text = f"{text}."

    text = text[0].upper() + text[1:]

    text = text.replace('"', "")
    text = text.replace("„", "")
    text = text.replace("“", "")
    text = text.replace("”", "")
    text = text.replace("„", "")
    text = text.replace("‟", "")
    text = text.replace("‟", "")
    for char in [
        "(",
        ")",
        "*",
        "-",
        "/",
        ":",
        ";",
        "«",
        "»",
        "–",
        "‘",
        "’",
        "‚",
        "…",
        "‹",
        "›",
        "а",
        "з",
        "к",
        "о",
        "т",
        "ъ",
    ]:
        text = text.replace(char, "")
    text = text.replace("\xa0", " ")
    text = text.replace("ß", "s")
    text = text.replace("á", "a")
    text = text.replace("ä", "e")
    text = text.replace("í", "i")
    text = text.replace("x", "ks")
    text = text.replace("ö", "o")
    text = text.replace("ü", "u")
    text = text.replace("ō", "o")
    text = text.replace("ŕ", "r")
    text = text.replace("ś", "s")
    text = text.replace("ů", "u")
    text = text.replace("ý", "y")

    return text


found_wavs = 0
current_speaker_idx = 0


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def format_sample(i, o, cut_off=False):
    if cut_off:
        os.system(
            f"sox -v 0.99 {i} {o} channels 1 norm -1 rate {SAMPLE_RATE} silence 1 1 {SILENCE_THRESHOLD} reverse silence 1 1 {SILENCE_THRESHOLD} reverse"
        )  # >/dev/null 2>&1
    else:
        os.system(
            f"sox -v 0.99 {i} {o} channels 1 norm -1 rate {SAMPLE_RATE}"
        )  # >/dev/null 2>&1

def exclude_ds_store(directory):
    for speaker in os.listdir(f"{directory['path']}/trl"):
        if speaker == ".DS_Store":
            os.remove(f"{directory['path']}/trl/{dataset_directory}")
        
    for rec_session in os.listdir(f"{directory['path']}/trl/{speaker}"):
        if rec_session == ".DS_Store":
            os.remove(f"{directory['path']}/trl/{speaker}/{rec_session}")
    
    for spoken_text in os.listdir(f"{directory['path']}/trl/{speaker}/{rec_session}"):
        if spoken_text == ".DS_Store":
            os.remove(f"{directory['path']}/trl/{speaker}/{rec_session}/{spoken_text}")

for directory in DATA_DIRS:
    if directory["type"] == "zalozba_film":
        exclude_ds_store(directory)


for directory in DATA_DIRS:
    final_metadata = ""
    if directory["type"] == "ljspeech":
        with open(f"{directory['path']}/metadata.csv", "r") as f:
            lj_metadata = f.read()
        current_speaker_idx += 1
        os.mkdir(f"{EXPORT_TO}/txt/{current_speaker_idx}")
        os.mkdir(f"{EXPORT_TO}/wav48/{current_speaker_idx}")
        with alive_bar(len(lj_metadata.splitlines())) as bar:
            for entry in lj_metadata.splitlines():
                found_wavs = found_wavs + 1
                wav, text, _ = entry.split("|")
                if directory["full_filename"]:
                    wav_path = f"{directory['path']}/wavs/{wav}"
                else:
                    wav_path = f"{directory['path']}/wavs/{wav}.wav"
                if not os.path.isfile(wav_path):
                    print(f" > Warning: {wav_path} does not exist, skipping!")
                    bar()
                else:
                    if not has_numbers(text):
                        text = procces_text(text)
                        random_str = get_random_string(32)
                        if directory["remove_silence"]:
                            format_sample(
                                wav_path,
                                f"{EXPORT_TO}/wav48/{current_speaker_idx}/{random_str}.{AUDIO_FORMAT}",
                                cut_off=True,
                            )
                        else:
                            format_sample(
                                wav_path,
                                f"{EXPORT_TO}/wav48/{current_speaker_idx}/{random_str}.{AUDIO_FORMAT}",
                                cut_off=False,
                            )

                        with open(
                            f"{EXPORT_TO}/txt/{current_speaker_idx}/{random_str}.txt",
                            "w",
                        ) as f:
                            f.write(text)
                    else:
                        print(f" > Warning: {text} has numbers, skipping!")
                    bar()

    if directory["type"] == "zalozba_lampa":
        with alive_bar() as bar:
            for dataset_directory in os.listdir(f"{directory['path']}/trl"):
                if dataset_directory != ".DS_Store":
                    for speaker in os.listdir(
                        f"{directory['path']}/trl/{dataset_directory}/RECS"
                    ):
                        is_speaker_ignored = False
                        for ignore_speaker in directory["ignore_speakers"]:
                            if not (
                                dataset_directory == ignore_speaker[0]
                                and speaker == ignore_speaker[1]
                            ):
                                is_speaker_ignored = True
                        if not is_speaker_ignored:
                            current_speaker_idx += 1
                            os.mkdir(f"{EXPORT_TO}/txt/{current_speaker_idx}")
                            os.mkdir(f"{EXPORT_TO}/wav48/{current_speaker_idx}")
                            for spoken_text in os.listdir(
                                f"{directory['path']}/trl/{dataset_directory}/RECS/{speaker}"
                            ):
                                found_wavs = found_wavs + 1
                                with open(
                                    f"{directory['path']}/trl/{dataset_directory}/RECS/{speaker}/{spoken_text}",
                                    "r",
                                ) as f:
                                    text = (
                                        f.read().replace("\n", " ").lower().rstrip(" ")
                                    )
                                    wav_path = f"{directory['path']}/sig/{dataset_directory}/RECS/{speaker}/{spoken_text.split('.')[0]}.wav"
                                    if not os.path.isfile(wav_path):
                                        # print(
                                        #     f" > Warning: {wav_path} does not exist, skipping!"
                                        # )
                                        pass
                                    else:
                                        if not has_numbers(text):
                                            text = procces_text(text)
                                            random_str = get_random_string(32)
                                            if directory["remove_silence"]:
                                                format_sample(
                                                    wav_path,
                                                    f"{EXPORT_TO}/wav48/{current_speaker_idx}/{random_str}.{AUDIO_FORMAT}",
                                                    cut_off=True,
                                                )
                                            else:
                                                format_sample(
                                                    wav_path,
                                                    f"{EXPORT_TO}/wav48/{current_speaker_idx}/{random_str}.{AUDIO_FORMAT}",
                                                    cut_off=False,
                                                )

                                            with open(
                                                f"{EXPORT_TO}/txt/{current_speaker_idx}/{random_str}.txt",
                                                "w",
                                            ) as f:
                                                f.write(text)
                                        else:
                                            print(
                                                f" > Warning: {text} has numbers, skipping!"
                                            )
                                        bar()
                        else:
                            print(f" > Not proccessing {dataset_directory}/{speaker}!")
    if directory["type"] == "zalozba_film":
        with alive_bar() as bar:
            for speaker in os.listdir(f"{directory['path']}/trl"):
                current_speaker_idx += 1
                os.mkdir(f"{EXPORT_TO}/txt/{current_speaker_idx}")
                os.mkdir(f"{EXPORT_TO}/wav48/{current_speaker_idx}")
                for rec_session in os.listdir(f"{directory['path']}/trl/{speaker}"):
                    for spoken_text in os.listdir(
                        f"{directory['path']}/trl/{speaker}/{rec_session}"
                    ):
                        found_wavs = found_wavs + 1
                        with open(
                            f"{directory['path']}/trl/{speaker}/{rec_session}/{spoken_text}",
                            "r",
                        ) as f:
                            text = f.read().replace("\n", " ").lower().rstrip(" ")
                            wav_path = f"{directory['path']}/sig/{speaker}/{rec_session}/{spoken_text.split('.')[0]}.wav"
                            if not os.path.isfile(wav_path):
                                # print(f" > Warning: {wav_path} does not exist, skipping!")
                                pass
                            else:
                                text = procces_text(text)
                                random_str = get_random_string(32)
                                if not has_numbers(text):
                                    format_sample(
                                        wav_path,
                                        f"{EXPORT_TO}/wav48/{current_speaker_idx}/{random_str}.{AUDIO_FORMAT}",
                                    )
                                    with open(
                                        f"{EXPORT_TO}/txt/{current_speaker_idx}/{random_str}.txt",
                                        "w",
                                    ) as f:
                                        f.write(text)
                                else:
                                    print(f" > Warning: {text} has numbers, skipping!")

                                bar()
            current_speaker_idx += 1
