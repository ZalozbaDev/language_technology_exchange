#!/usr/bin/python3 

""" Created 2023

    @author: korla, tamas

Copyright (c) 2024 Korla Baier, Tamás Janusko

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
# pip3 install alive-progress
from alive_progress import alive_bar
# available from apt
import pandas as pd
import shutil

# TODO add all data directories here!
DATA_DIRS = [
{
        "path": "./resources/speech_corpus_weronika_recordings/",
        "type": "ljspeech",
},
{
	"path": "./resources/speech_corpus_2020_study_lamp/",
	"type": "zalozba_lampa",
},
{
	"path": "./resources/speech_corpus_film_9_a_pol/",
	"type": "zalozba_film",
},
{
	"path": "./resources/cv-corpus-18.0-2024-06-14/hsb/",
	"type": "cv",
},
{
	"path": "./resources/speech_corpus_film_gilles/",
	"type": "zalozba_film",
},
{
	"path": "./resources/speech_corpus_film_karla_a_katrina/",
	"type": "zalozba_film",
},
{
	"path": "./resources/speech_corpus_korla_recordings/dataset/",
	"type": "ljspeech",
},
{
	"path": "./resources/speech_corpus_film_mpz_insekten/",
	"type": "zalozba_film",
},
{
	"path": "./resources/speech_corpus_film_mpz_reise/",
	"type": "zalozba_film",
},
{
	"path": "./resources/speech_corpus_film_mpz_wjedro/",
	"type": "zalozba_film",
},
{
	"path": "./resources/speech_corpus_film_peeweeje/",
	"type": "zalozba_film",
},
{
	"path": "./resources/speech_corpus_film_syn_winnetouwa/",
	"type": "zalozba_film",
},
{
        "path": "./resources/speech_corpus_mic_recordings_1/",
        "type": "micspeech",
},
{
        "path": "./resources/speech_corpus_mic_recordings_2/",
        "type": "micspeech",
},
{
        "path": "./resources/speech_corpus_mic_recordings_3/",
        "type": "micspeech",
},
{
        "path": "./resources/speech_corpus_mic_recordings_4/",
        "type": "micspeech",
},
{
        "path": "./resources/speech_corpus_mic_recordings_5/",
        "type": "micspeech",
},
{
        "path": "./resources/speech_corpus_mic_recordings_6/",
        "type": "micspeech",
},
{
        "path": "./resources/speech_corpus_film_mala_wjera/",
        "type": "micspeech",
},
{
        "path": "./resources/speech_corpus_film_sonina_desca/",
        "type": "micspeech",
},

]

EXPORT_AS = "hf"
EXPORT_TO = "./export"

SAMPLE_RATE = 16_000
AUDIO_FORMAT = "wav"

LOWERCASE = True

def get_random_string(length):
	letters = string.ascii_lowercase
	return "".join(random.choice(letters) for _ in range(length))
	
if os.path.exists(f"{EXPORT_TO}/wavs"):
	shutil.rmtree(f"{EXPORT_TO}/wavs")
os.makedirs(f"{EXPORT_TO}/wavs")
	
with open(f"{EXPORT_TO}/metadata.csv", "w") as f:
	f.write("file_name,transcription")
	
def procces_text(text: str):
	if LOWERCASE:
		text = text.lower()
		
	if not (text[-1] == "." or text[-1] == "!" or text[-1] == "?"):
		text = f"{text}."
	
	# disabled - do not capitalize first letter
	# text = text[0].upper() + text[1:]

	text = text.replace('"', "")
	text = text.replace("„", "")
	text = text.replace("“", "")
	text = text.replace("”", "")
	text = text.replace("„", "")
	text = text.replace("‟", "")

	return text

for directory in DATA_DIRS:
	
	final_metadata = ""
	
	print ("Procesing: ", directory['path'])
	
	if directory["type"] == "ljspeech":
		with open(f"{directory['path']}/metadata.csv", "r") as f:
			lj_metadata = f.read()
		with alive_bar(len(lj_metadata.splitlines())) as bar:
			for entry in lj_metadata.splitlines():
				wav, text, _ = entry.split("|")
				wav_path = f"{directory['path']}/wavs/{wav}.wav"
				if not os.path.isfile(wav_path):
					print(f" > Warning: {wav_path} does not exist, skipping!")
					bar()
				else:
					print()
					text = procces_text(text)
					random_str = get_random_string(32)
					os.system(
						f"ffmpeg -i {wav_path} -ar {SAMPLE_RATE} -vn -ac 1 {EXPORT_TO}/wavs/{random_str}.{AUDIO_FORMAT} >/dev/null 2>&1"
					)
					if EXPORT_AS == "hf":
						final_metadata = (
							f'{final_metadata}\nwavs/{random_str}.wav,"{text}"'
						)
					else:
						final_metadata = f"{final_metadata}\n{random_str}|{text}|{text}"
					bar()	
	
	if directory["type"] == "zalozba_lampa":
		with alive_bar() as bar:
			for dataset_directory in os.listdir(f"{directory['path']}/trl"):
				for speaker in os.listdir(
					f"{directory['path']}/trl/{dataset_directory}/RECS"
				):
					for spoken_text in os.listdir(
						f"{directory['path']}/trl/{dataset_directory}/RECS/{speaker}"
					):
						with open(
							f"{directory['path']}/trl/{dataset_directory}/RECS/{speaker}/{spoken_text}",
							"r",
						) as f:
							text = f.read().replace("\n", " ").lower().rstrip(" ")
							wav_path = f"{directory['path']}/sig/{dataset_directory}/RECS/{speaker}/{spoken_text.split('.')[0]}.wav"
							if not os.path.isfile(wav_path):
								# print(
								#	f" > Warning: {wav_path} does not exist, skipping!"
								# )
								pass
							else:
								text = procces_text(text)
								random_str = get_random_string(32)
								os.system(
									f"ffmpeg -i {wav_path} -ar {SAMPLE_RATE} -vn -ac 1 {EXPORT_TO}/wavs/{random_str}.{AUDIO_FORMAT} >/dev/null 2>&1" # >/dev/null 2>&1
								)
								if EXPORT_AS == "hf":
									final_metadata = f'{final_metadata}\nwavs/{random_str}.wav,"{text}"'
								else:
									final_metadata = (
										f"{final_metadata}\n{random_str}|{text}|{text}"
									)
								bar()

	if directory["type"] == "zalozba_film":
		with alive_bar() as bar:
			for speaker in os.listdir(f"{directory['path']}/trl"):
				for rec_session in os.listdir(f"{directory['path']}/trl/{speaker}"):
					for spoken_text in os.listdir(
						f"{directory['path']}/trl/{speaker}/{rec_session}"
					):
						with open(
							f"{directory['path']}/trl/{speaker}/{rec_session}/{spoken_text}",
							"r",
						) as f:
							text = f.read().replace("\n", " ").lower().rstrip(" ")
							wav_path = f"{directory['path']}/sig/{speaker}/{rec_session}/{spoken_text.split('.')[0]}.wav"
							if not os.path.isfile(wav_path):
								# print(
								#	f" > Warning: {wav_path} does not exist, skipping!"
								# )
								pass
							else:
								text = procces_text(text)
								random_str = get_random_string(32)
								os.system(
									f"ffmpeg -i {wav_path} -ar {SAMPLE_RATE} -vn -ac 1 {EXPORT_TO}/wavs/{random_str}.{AUDIO_FORMAT} >/dev/null 2>&1" # >/dev/null 2>&1
								)
								if EXPORT_AS == "hf":
									final_metadata = (
										f'{final_metadata}\nwavs/{random_str}.wav,"{text}"'
									)
								else:
									final_metadata = (
										f"{final_metadata}\n{random_str}|{text}|{text}"
									)
								bar()

	if directory["type"] == "cv":
		data = pd.read_csv(f"{directory['path']}/validated.tsv", sep="\t")
		data = data.drop(
			columns=[
				"client_id",
				"up_votes",
				"down_votes",
				"age",
				"gender",
				"accents",
				"locale",
				"segment",
			]
		)
		with alive_bar(len(list(data.iterrows()))) as bar:
			for row in data.itertuples():
				path = f"{directory['path']}/clips/{row.__getattribute__('path')}"
				sentence = row.__getattribute__("sentence")
				if not os.path.isfile(path):
					print(f" > Warning: {path} does not exist, skipping!")
					pass
				else:
					sentence = procces_text(sentence)
					random_str = get_random_string(32)
					os.system(
						f"ffmpeg -i {path} -ar {SAMPLE_RATE} -vn -ac 1 {EXPORT_TO}/wavs/{random_str}.{AUDIO_FORMAT} >/dev/null 2>&1" # >/dev/null 2>&1
					)
					if EXPORT_AS == "hf":
						final_metadata = (
							f'{final_metadata}\nwavs/{random_str}.wav,"{sentence}"'
						)
					else:
						final_metadata = (
							f"{final_metadata}\n{random_str}|{sentence}|{sentence}"
							)
				bar()


	if directory["type"] == "micspeech":
		with alive_bar() as bar:
			for speaker in os.listdir(f"{directory['path']}/trl"):
				for rec_session in os.listdir(f"{directory['path']}/trl/{speaker}"):
					for spoken_text in os.listdir(
						f"{directory['path']}/trl/{speaker}/{rec_session}"
					):
						with open(
							f"{directory['path']}/trl/{speaker}/{rec_session}/{spoken_text}",
							"r",
						) as f:
							text = f.read().replace("\n", " ").lower().rstrip(" ")
							wav_path = f"{directory['path']}/sig/{speaker}/{rec_session}/{spoken_text.split('.')[0]}.wav"
							if not os.path.isfile(wav_path):
								# print(
								#	f" > Warning: {wav_path} does not exist, skipping!"
								# )
								pass
							else:
								text = procces_text(text)
								random_str = get_random_string(32)
								os.system(
									f"ffmpeg -i {wav_path} -ar {SAMPLE_RATE} -vn -ac 1 {EXPORT_TO}/wavs/{random_str}.{AUDIO_FORMAT}___MIC >/dev/null 2>&1" # >/dev/null 2>&1
								)
								if EXPORT_AS == "hf":
									final_metadata = (
										f'{final_metadata}\nwavs/{random_str}.wav,"{text}"  MIC _________'
									)
								else:
									final_metadata = (
										f"{final_metadata}\n{random_str}|{text}|{text} MIC _________"
									)
								bar()



	with open(f"{EXPORT_TO}/metadata.csv", "a") as f:
		f.write(final_metadata)
