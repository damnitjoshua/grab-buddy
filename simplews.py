from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf
import io
import logging

app = FastAPI()

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)


@app.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection accepted.")
    try:
        while True:
            # Receive raw Float32 audio buffer from frontend
            data = await websocket.receive_bytes()
            audio_np = np.frombuffer(data, dtype=np.float32)

            # Optional: process audio here (e.g., TTS, noise reduction, etc.)
            # For now, just echo the audio back

            # Convert to WAV in memory
            with io.BytesIO() as wav_io:
                sf.write(wav_io, audio_np, 44100, format='WAV')
                wav_io.seek(0)
                wav_bytes = wav_io.read()

            # Send WAV audio back to frontend
            await websocket.send_bytes(wav_bytes)

    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()
        logging.info("WebSocket connection closed.")
