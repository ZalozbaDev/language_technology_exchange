import json
import os
import random
import shutil
import string
import subprocess
import time
from typing import List, Optional, Tuple

import pandas as pd

from common import ProcessingState


def calculate_total_files(data_dirs):
    """Calculate the total number of files in all directories"""
    total = 0
    for directory, dir_type in data_dirs:
        total += count_total_files(directory, dir_type)
    return total


def print_progress(file_path, current_overall, total_overall):
    """Output progress information in JSON format"""
    progress = {
        "file": file_path,
        "current": current_overall,
        "total": total_overall,
        "progress": (current_overall / total_overall * 100) if total_overall > 0 else 0,
    }
    print(json.dumps(progress, ensure_ascii=False))


class PreprocessingConfig:
    def __init__(
        self,
        data_dirs: List[Tuple[str, str]],
        export_as: str = "hf",
        export_to: str = "./TrainingData/export",
        sample_rate: int = 16000,
        audio_format: str = "wav",
        lowercase: bool = True,
    ):
        self.data_dirs = data_dirs
        self.export_as = export_as
        self.export_to = export_to
        self.sample_rate = sample_rate
        self.audio_format = audio_format
        self.lowercase = lowercase


class PreprocessingCallback:
    def on_progress(self, file_path: str, current: int, total: int):
        pass

    def on_status(self, message: str):
        pass

    def on_error(self, error: str):
        pass


def get_random_string(length):
    """Generate a random string"""
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def process_text(text: str, lowercase=True) -> str:
    """Process text"""
    if lowercase:
        text = text.lower()

    if not text.endswith((".", "!", "?")):
        text += "."

    for char in ['"', "„", '"', '"', "‟"]:
        text = text.replace(char, "")

    return text


def convert_audio(
    input_path: str, output_path: str, sample_rate: int, state: ProcessingState
) -> bool:
    """Convert audio file with cancellation support"""
    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-ar",
            str(sample_rate),
            "-vn",
            "-ac",
            "1",
            output_path,
        ]
        process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        state.add_process(process)

        while True:
            if state.should_stop:
                process.terminate()
                process.wait()
                state.remove_process(process)
                return False

            ret = process.poll()
            if ret is not None:
                break
            time.sleep(0.1)

        state.remove_process(process)
        if ret == 0:
            return True
        else:
            state.add_log(f"Error processing {input_path}")
            return False
    except Exception as e:
        try:
            state.remove_process(process)
        except:
            pass  # 如果 process 不在列表中，忽略
        state.add_log(f"Error processing {input_path}: {str(e)}")
        return False


def count_total_files(directory, dir_type):
    """Calculate the total number of files in a directory based on its type"""
    total = 0
    try:
        if dir_type == "ljspeech":
            with open(f"{directory}/metadata.csv", "r", encoding="utf-8") as f:
                total = sum(1 for _ in f)
        elif dir_type == "zalozba_lampa":
            for dataset_directory in os.listdir(f"{directory}/trl"):
                for speaker in os.listdir(f"{directory}/trl/{dataset_directory}/RECS"):
                    total += len(
                        os.listdir(
                            f"{directory}/trl/{dataset_directory}/RECS/{speaker}"
                        )
                    )
        elif dir_type == "zalozba_film":
            for speaker in os.listdir(f"{directory}/trl"):
                for rec_session in os.listdir(f"{directory}/trl/{speaker}"):
                    total += len(os.listdir(f"{directory}/trl/{speaker}/{rec_session}"))
        elif dir_type == "cv":
            data = pd.read_csv(f"{directory}/validated.tsv", sep="\t")
            total = len(data)
    except Exception as e:
        print(json.dumps({"error": f"Error counting files: {str(e)}"}))
        return 0
    return total


def process_ljspeech(
    directory: str,
    export_to: str,
    sample_rate: int,
    audio_format: str,
    export_as: str,
    lowercase: bool,
    current_overall: int,
    total_overall: int,
    callback: Optional[PreprocessingCallback],
    state: ProcessingState,
) -> str:
    """Process LJSpeech format data"""
    final_metadata = ""
    processed = 0
    total_files = count_total_files(directory, "ljspeech")

    try:
        with open(f"{directory}/metadata.csv", "r", encoding="utf-8") as f:
            for line in f:
                wav, text, _ = line.strip().split("|")
                wav_path = f"{directory}/wavs/{wav}.wav"

                if os.path.isfile(wav_path):
                    text = process_text(text, lowercase)
                    random_str = get_random_string(32)
                    output_path = f"{export_to}/wavs/{random_str}.{audio_format}"

                    if convert_audio(wav_path, output_path, sample_rate, state):
                        if export_as == "hf":
                            final_metadata += f'\nwavs/{random_str}.wav,"{text}"'
                        else:
                            final_metadata += f"\n{random_str}|{text}|{text}"

                processed += 1
                if callback:
                    callback.on_progress(
                        wav_path, current_overall + processed, total_overall
                    )

    except Exception as e:
        if callback:
            callback.on_error(f"Error processing LJSpeech data: {str(e)}")

    return final_metadata.strip()


def process_zalozba_film(
    directory,
    export_to,
    sample_rate,
    audio_format,
    export_as,
    lowercase,
    current_overall,
    total_overall,
    callback: Optional[PreprocessingCallback],
    state: ProcessingState,
):
    """Process Zalozba Film format data"""
    final_metadata = ""
    processed = 0
    total_files = count_total_files(directory, "zalozba_film")

    try:
        for speaker in os.listdir(f"{directory}/trl"):
            for rec_session in os.listdir(f"{directory}/trl/{speaker}"):
                for spoken_text in os.listdir(
                    f"{directory}/trl/{speaker}/{rec_session}"
                ):
                    transcript_file = (
                        f"{directory}/trl/{speaker}/{rec_session}/{spoken_text}"
                    )
                    sig_path = f"{directory}/sig/{speaker}/{rec_session}/{spoken_text.split('.')[0]}.wav"

                    if os.path.isfile(sig_path):
                        with open(transcript_file, "r", encoding="utf-8") as f_trans:
                            text = process_text(f_trans.read(), lowercase)

                        random_str = get_random_string(32)
                        output_wav = f"{export_to}/wavs/{random_str}.{audio_format}"

                        if convert_audio(sig_path, output_wav, sample_rate, state):
                            if export_as == "hf":
                                final_metadata += f'\nwavs/{random_str}.wav,"{text}"'
                            else:
                                final_metadata += f"\n{random_str}|{text}|{text}"

                    processed += 1
                    if callback:
                        callback.on_progress(
                            sig_path, current_overall + processed, total_overall
                        )

    except Exception as e:
        if callback:
            callback.on_error(f"Error processing Zalozba Film data: {str(e)}")

    return final_metadata.strip()


def process_zalozba_lampa(
    directory,
    export_to,
    sample_rate,
    audio_format,
    export_as,
    lowercase,
    current_overall,
    total_overall,
    callback: Optional[PreprocessingCallback],
    state: ProcessingState,
):
    """Process Zalozba Lampa format data"""
    final_metadata = ""
    processed = 0
    total_files = count_total_files(directory, "zalozba_lampa")

    try:
        for dataset_directory in os.listdir(f"{directory}/trl"):
            for speaker in os.listdir(f"{directory}/trl/{dataset_directory}/RECS"):
                for spoken_text in os.listdir(
                    f"{directory}/trl/{dataset_directory}/RECS/{speaker}"
                ):
                    transcript_path = f"{directory}/trl/{dataset_directory}/RECS/{speaker}/{spoken_text}"
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        text = f.read().replace("\n", " ").rstrip(" ")
                        wav_path = f"{directory}/sig/{dataset_directory}/RECS/{speaker}/{spoken_text.split('.')[0]}.wav"

                        if os.path.isfile(wav_path):
                            text = process_text(text, lowercase)
                            random_str = get_random_string(32)
                            output_wav = f"{export_to}/wavs/{random_str}.{audio_format}"

                            if convert_audio(wav_path, output_wav, sample_rate, state):
                                if export_as == "hf":
                                    final_metadata += (
                                        f'\nwavs/{random_str}.wav,"{text}"'
                                    )
                                else:
                                    final_metadata += f"\n{random_str}|{text}|{text}"

                    processed += 1
                    if callback:
                        callback.on_progress(
                            wav_path, current_overall + processed, total_overall
                        )

    except Exception as e:
        if callback:
            callback.on_error(f"Error processing Zalozba Lampa data: {str(e)}")

    return final_metadata.strip()


def process_cv(
    directory,
    export_to,
    sample_rate,
    audio_format,
    export_as,
    lowercase,
    current_overall,
    total_overall,
    callback: Optional[PreprocessingCallback],
    state: ProcessingState,
):
    """Process Common Voice format data"""
    final_metadata = ""
    processed = 0

    try:
        data = pd.read_csv(f"{directory}/validated.tsv", sep="\t")
        total_files = len(data)

        for _, row in data.iterrows():
            path = f"{directory}/clips/{row['path']}"
            sentence = row["sentence"]

            if os.path.isfile(path):
                sentence = process_text(sentence, lowercase)
                random_str = get_random_string(32)
                output_wav = f"{export_to}/wavs/{random_str}.{audio_format}"

                if convert_audio(path, output_wav, sample_rate, state):
                    if export_as == "hf":
                        final_metadata += f'\nwavs/{random_str}.wav,"{sentence}"'
                    else:
                        final_metadata += f"\n{random_str}|{sentence}|{sentence}"

            processed += 1
            if callback:
                callback.on_progress(path, current_overall + processed, total_overall)

    except Exception as e:
        if callback:
            callback.on_error(f"Error processing Common Voice data: {str(e)}")

    return final_metadata.strip()


def process_directory(
    directory_path: str,
    dir_type: str,
    export_to: str,
    sample_rate: int,
    audio_format: str,
    export_as: str,
    lowercase: bool,
    current_overall: int,
    total_overall: int,
    callback: Optional[PreprocessingCallback],
    state: ProcessingState,
) -> str:
    """Process a single directory"""
    if callback:
        callback.on_status(
            f"Starting to process directory: {directory_path} ({dir_type})"
        )

    processors = {
        "ljspeech": process_ljspeech,
        "zalozba_film": process_zalozba_film,
        "zalozba_lampa": process_zalozba_lampa,
        "cv": process_cv,
    }

    if dir_type in processors:
        try:
            return processors[dir_type](
                directory_path,
                export_to,
                sample_rate,
                audio_format,
                export_as,
                lowercase,
                current_overall,
                total_overall,
                callback,
                state,
            )
        except Exception as e:
            if callback:
                callback.on_error(f"Error processing {dir_type} data: {str(e)}")
            return ""
    else:
        if callback:
            callback.on_error(f"Unknown directory type: {dir_type}")
        return ""


def preprocess_dataset(
    config: PreprocessingConfig, callback: Optional[PreprocessingCallback]
) -> bool:
    state = callback.state if callback and hasattr(callback, "state") else None

    try:
        if os.path.exists(f"{config.export_to}/wavs"):
            shutil.rmtree(f"{config.export_to}/wavs")
        os.makedirs(f"{config.export_to}/wavs", exist_ok=True)

        metadata_path = f"{config.export_to}/metadata.csv"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("file_name,transcription\n")

        total_files = calculate_total_files(config.data_dirs)
        processed_files = 0

        if callback:
            callback.on_status(f"Total files: {total_files}")

        for path, dir_type in config.data_dirs:
            dir_total = count_total_files(path, dir_type)
            if callback:
                callback.on_status(
                    f"Processing directory: {path} ({dir_type}), Files: {dir_total}"
                )

            metadata = process_directory(
                path,
                dir_type,
                config.export_to,
                config.sample_rate,
                config.audio_format,
                config.export_as,
                config.lowercase,
                processed_files,
                total_files,
                callback,
                state,
            )

            if metadata:
                with open(metadata_path, "a", encoding="utf-8") as f:
                    f.write(metadata + "\n")

            processed_files += dir_total
            if callback:
                callback.on_progress("", processed_files, total_files)

        if callback:
            callback.on_status("Processing completed")
        return True

    except Exception as e:
        if callback:
            callback.on_error(f"Processing failed: {str(e)}")
        return False
