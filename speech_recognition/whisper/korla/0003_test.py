#!/usr/bin/python3 

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


from transformers import (
	WhisperProcessor,
	WhisperForConditionalGeneration,
	WhisperTokenizer,
	WhisperFeatureExtractor
)
import gradio as gr
import librosa
import torch

from transformers import pipeline

# define model to change
model_id = "openai/whisper-small"
# model_id = "/home/danielzoba/hsb_stt_demo/hsb_whisper/"

processor = WhisperProcessor.from_pretrained(model_id)


# adjust path 
# model = WhisperForConditionalGeneration.from_pretrained("./training/results/checkpoint-500/")
model = WhisperForConditionalGeneration.from_pretrained("/home/danielzoba/hsb_stt_demo/hsb_whisper")
model.config.forced_decoder_ids = None

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)

tokenizer = WhisperTokenizer.from_pretrained(model_id, task="transcribe")

pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

if torch.cuda.is_available():
	device = torch.device("cuda")
elif (
	hasattr(torch.backends, "mps")
	and torch.backends.mps.is_available()
	and torch.backends.mps.is_built()
):
	device = torch.device("mps")
else:
	device = torch.device("cpu")
	
model.to(device)

def transcribe(audio):
	sample = librosa.load(audio, sr=16_000, mono=True)[0]
	
	input_features = processor(
		sample, sampling_rate=16_000, return_tensors="pt"
	).input_features
	
	input_features = input_features.to(device)
	
	predicted_ids = model.generate(input_features)
	
	transcription_whspr = processor.batch_decode(
		predicted_ids, skip_special_tokens=True
	)[0]
	
	return transcription_whspr

def transcribe_file(filepath):
    output = pipe(
        filepath,
        max_new_tokens=256,
        generate_kwargs={
            "task": "transcribe",
            # "language": "english",
        },  # update with the language you've fine-tuned on
        chunk_length_s=30,
        batch_size=8,
    )
    return output["text"]

iface = gr.Interface(
	fn=transcribe,
	inputs=gr.Audio(source="microphone", type="filepath"),
	outputs="text",
	title="Serbski STT",
	description="Gradio demo za spóznawanje rěće w hornjoserbšćinje",
)

file_transcribe = gr.Interface(
    fn=transcribe_file,
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs=gr.outputs.Textbox(),
)

demo = gr.Blocks()

with demo:
    gr.TabbedInterface(
        [iface, file_transcribe],
        ["Transcribe Microphone", "Transcribe Audio File"],
    )

demo.launch(debug=True,share=False,server_name="0.0.0.0",ssl_verify=False,ssl_keyfile="domain.key",ssl_certfile="domain.crt")
