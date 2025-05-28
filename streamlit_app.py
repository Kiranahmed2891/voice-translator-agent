
#step 1 create folder (folder name)
#Step 2: Create Virtual Environment   #command: python -m venv .venv 
 #activate it window command : .venv\Scripts\activate
#step 3  Install Required Packages
# commands :  pip install streamlit gTTS SpeechRecognition sounddevice scipy python-dotenv
# Step 4: Setup .env File 
#  GEMINI_API_KEY=your_gemini_api_key_here
#Step 6: Add Your Code
 # Paste your full code into a file named main.py.
#Step 7: Run the App
#From the terminal (make sure .venv is activated):
# command:  streamlit run main.py







import streamlit as st
import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wav
from gtts import gTTS
import os
import tempfile
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv

# Load API Key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Setup model
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Translator Agent
translator = Agent(
    name="Translator Agent",
    instructions="""
    You are a translator agent. Your job is to translate any input text between English and Urdu.
    If the input is in English, translate it into Urdu.
    If the input is in Urdu, translate it into English.
    Be precise and accurate.
    """
)

async def translate_text(text):
    response = await Runner.run(translator, input=text, run_config=config)
    return response.final_output

# Voice recording
def record_audio(duration=5, fs=16000):
    st.info(f"🎙 Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_audio.name, fs, recording)
    return temp_audio.name

def recognize_speech(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language='ur-PK')  # Change language if needed
    except sr.UnknownValueError:
        return "Sorry, couldn't understand."
    except sr.RequestError:
        return "Speech recognition failed."

def speak(text, lang='ur'):
    tts = gTTS(text=text, lang=lang)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    os.system(f"start {temp_file.name}")  # For Windows

# Streamlit UI
st.title("🎙 Urdu ↔ English Voice Translator")

option = st.selectbox("Choose an action", ["Type Text", "Speak"])

if option == "Type Text":
    user_input = st.text_area("Enter text here:")
    if st.button("Translate"):
        if user_input.strip():
            with st.spinner("Translating..."):
                result = asyncio.run(translate_text(user_input))
                st.success("Translation:")
                st.write(result)
                speak(result, lang='en' if 'ا' in result or 'ی' in result else 'ur')
        else:
            st.warning("Please enter some text.")
else:
    if st.button("🎤 Start Voice Input"):
        audio_path = record_audio()
        spoken_text = recognize_speech(audio_path)
        st.info(f"Recognized: {spoken_text}")
        if spoken_text:
            with st.spinner("Translating..."):
                result = asyncio.run(translate_text(spoken_text))
                st.success("Translation:")
                st.write(result)
                speak(result, lang='en' if 'ا' in result or 'ی' in result else 'ur')       
