import requests
import streamlit as st

def gerar_audio_openai(texto):
    url = "https://api.openai.com/v1/audio/speech"
    
    headers = {
        "Authorization": f"Bearer {st.secrets['OPENAI']['OPENAI_API_KEY']}", 
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "tts-1",  # ou "tts-1-hd" para melhor qualidade
        "input": texto,
        "voice": "alloy",
        "response_format": "mp3",  # mp3, opus, aac, flac, wav, pcm
        "speed": 1.0  # 0.25 a 4.0
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    return response