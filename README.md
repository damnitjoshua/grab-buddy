#  Grab Buddy

Grab Buddy is a voice-first AI assistant designed to streamline the workflow of Grab drivers, enhancing their efficiency and safety on the road

Check out our technical documentation [here](https://docs.google.com/document/d/1rJnp1w4t5gLsYAt71yyAbK3h2b3cDaPSX-DbTpzocus/edit?usp=sharing).
Check out our slides [here](https://docs.google.com/presentation/d/1UNVEqKLw4onzt-WcmKoE9auZqQljWi289nAipNrMeS8/edit?usp=sharing).


## 🏗️ Solution Architecture

Our system enables seamless voice interaction for drivers through the following pipeline:

1. Real-Time Audio Streaming
Audio chunks are streamed from the Next.js frontend to a FastAPI Python backend via WebSockets.


2. Noise Suppression
Incoming audio is denoised using the noisereduce library to improve transcription accuracy.

3. Speech-to-Text Conversion
The denoised audio is transcribed using Mesolitica STT (300M model), supporting multilingual input.

4.  Intent Recognition
Transcribed text is analyzed by Gemini Flash 2.0 Lite to identify the driver's intent and trigger appropriate actions.

5. Text-to-Speech Feedback
Audio responses are generated using Kokoro TTS (86M model) for natural and responsive feedback.

6. Data Storage & Analytics
All interactions are stored in a Supabase SQL database for monitoring, analytics, and performance insights.



## 🚀 Built with

- NextJS
- WebSockets
- FastAPI
- Streamlit
- Supabase
- Hugging Face
- Gemini


## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/damnitjoshua/grab-buddy

# Navigate into the directory
cd grab-buddy
```



## ⚙️ Setup

### 1. Start the NextJS Frontend

```bash
# Navigate into frontend folder
cd frontend

# Install dependencies
npm install

# Run the dev server
npm run dev
```

### 2. Start the Python Backend

```bash
# Setup Python virtual environment (Use Python3.12)
python -m venv venv

# Activate the virtual environment (MacOS)
source venv/bin/activate 

# Activate the virtual environment (Windows)
venv/Scripts/activate

# Install the dependencies
pip install -r requirements.txt

# Start the FastAPI server
fastapi dev speechModule2-webs.py
```

### 3. Start the Analytics dashboard

```bash
# Navigate into analytics folder
cd analytics

# Start the streamlit application
streamlit run app.py
```

***

🚗😊 Happy Driving!

