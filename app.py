import os
from seamless_communication.models.inference.translator import Translator
import torchaudio
import torch
import gradio as gr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
translator = Translator(
    model_name_or_card="seamlessM4T_large",
    vocoder_name_or_card="vocoder_36langs",
    device=device,
    dtype=None,
)

def speech_to_text(path_to_input_audio):
  """
  for language abbreviation visit : https://github.com/facebookresearch/seamless_communication/tree/main/scripts/m4t/predict
  """
  resample_rate = 16000
  waveform, sample_rate = torchaudio.load(path_to_input_audio)
  resampler = torchaudio.transforms.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
  resampled_waveform = resampler(waveform)
  torchaudio.save("/content/input.wav", resampled_waveform, resample_rate)
  translated_text, _, _ = translator.predict("/content/input.wav", "s2tt", "eng")
  return str(translated_text)
    
def eng_text_to_arb(input_text):
  """
  for language abbreviation visit : https://github.com/facebookresearch/seamless_communication/tree/main/scripts/m4t/predict
  """
  translated_text, wav, sr = translator.predict(input_text, "t2st", "arb", src_lang="eng")
  return str(translated_text)

demo = gr.Interface(
    speech_to_text,
    title="Speech-to-text",
    inputs= gr.Audio(type = "filepath"),
    outputs = "text"
)

demo.launch()
