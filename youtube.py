"""
This module provides functionality to process YouTube videos by downloading their audio, 
transcribing the audio using the Whisper model, translating the transcription to English, 
and saving the translated text to a file.

Key Features:
- Download audio from a YouTube video.
- Transcribe audio using the Whisper model.
- Translate the transcription to English using Google Translate.
- Save the translated text to a file in the 'uploads' folder.
"""

import os
from pytubefix import YouTube
import whisper
import warnings
import torch
from googletrans import Translator
import asyncio
import re

class YouTubeProcessor:
    def __init__(self, url):
        self.url = url
        self.yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
        self.audio_file = self.sanitize_filename(self.yt.title) + ".m4a"
        self.txt_file = self.sanitize_filename(self.yt.title) + ".txt"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model("base").to(self.device)
        self.upload_folder = "uploads"
        os.makedirs(self.upload_folder, exist_ok=True)

    def sanitize_filename(self, filename):
        """
        Sanitizes a filename by removing invalid characters.

        Args:
            filename (str): The original filename.

        Returns:
            str: The sanitized filename.
        """
        return re.sub(r'[\\/*?:"<>|]', "", filename)

    def download_audio(self):
        """
        Downloads the audio stream of the YouTube video.
        """
        audio_stream = self.yt.streams.filter(only_audio=True).first()
        audio_stream.download(filename=self.audio_file)
        print("Audio download complete!")

    def transcribe_audio(self):
        """
        Transcribes the downloaded audio using the Whisper model.

        Returns:
            str: The transcribed text.
        """
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
        result = self.model.transcribe(self.audio_file)
        transcribed_text = result["text"]
        return transcribed_text

    async def translate_text(self, text):
        """
        Translates the given text to English.

        Args:
            text (str): The text to translate.

        Returns:
            str: The translated text.
        """
        translator = Translator()
        translated = await translator.translate(text, src='auto', dest='en')
        return translated.text

    def save_translated_text(self, translated_text):
        """
        Saves the translated text to a file in the uploads folder.

        Args:
            translated_text (str): The translated text to save.
        """
        txt_file_path = os.path.join(self.upload_folder, self.txt_file)
        with open(txt_file_path, "w", encoding="utf-8") as file:
            file.write(translated_text)
        print(f"Translated text saved to {txt_file_path}")

    def process(self):
        """
        Processes the YouTube video by downloading audio, transcribing it, translating the transcription, 
        and saving the translated text.

        Returns:
            str: The name of the saved text file.
        """
        self.download_audio()
        transcribed_text = self.transcribe_audio()
        translated_text = asyncio.run(self.translate_text(transcribed_text))
        self.save_translated_text(translated_text)
        return self.txt_file
