from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import speech_recognition as sr
import google.generativeai as genai
from typing import List, Dict, Optional, Any
import json
from kokoro import KPipeline
import sounddevice as sd
import numpy as np
from transformers import AutoProcessor, AutoModelForCTC
import sys
import noisereduce as nr
import logging
import os
import threading
import time
import torch  # Import torch
import warnings  # Import warnings module
import asyncio
import io
import soundfile as sf  # Import soundfile for saving audio chunks
from supabase import create_client, Client
import uuid  # Import uuid for generating unique driver IDs

app = FastAPI()

# Enable CORS for local development (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - adjust for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Keys and Model Settings ---
# Replace with your actual API key or environment variable
GOOGLE_API_KEY = "AIzaSyAf8HxoCsImAlgliS43dTaixkIbC6mgR7o"
GEMINI_MODEL_NAME = "gemini-2.0-flash-lite"
GEMINI_TEMPERATURE = 0.3
GEMINI_TOP_P = 0.9
GEMINI_TOP_K = 20
GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

WHISPER_MODEL_ID = "mesolitica/wav2vec2-xls-r-300m-mixed"
KOKORO_VOICE = 'bf_emma'
KOKORO_SAMPLERATE = 24000
NOISE_REDUCE_PROP_DECREASE = 0.9  # You can tune this parameter
NOISE_REDUCE_TIME_CONSTANT_S = 0.5  # You can tune this parameter
SPEECH_RECOGNITION_TIMEOUT = 10  # Increased speech recognition timeout
TTS_DELAY = 0.5  # Delay after TTS before listening again
KOKORO_REPO_ID = 'hexgrad/Kokoro-82M'  # Define Kokoro repo_id
# Maximum consecutive unknown intents before fallback
MAX_CONSECUTIVE_UNKNOWN_INTENTS = 3

USE_GEMINI = bool(GOOGLE_API_KEY)
tts_pipeline = None
whisper_processor = None
whisper_model = None
chat_history_global = []  # Global chat history for WebSocket sessions
tts_stop_event = threading.Event()  # Event to stop TTS
tts_playback_thread = None  # Thread for TTS playback
# sample_rides = ["Ride to Sunway Universiti from Universiti Tower, RM10",
#                "Ride to KLIA from KLCC, RM60", "Ride to Monash from Sunway Universiti, RM35"]
last_chosen_destination_global = None
consecutive_unknown_intents_global = 0
tts_lock = asyncio.Lock()  # Async lock to control TTS access

# --- Initialize Gemini ---
if USE_GEMINI:
    genai.configure(api_key=GOOGLE_API_KEY)
    generation_config = genai.GenerationConfig(
        temperature=GEMINI_TEMPERATURE,
        top_p=GEMINI_TOP_P,
        top_k=GEMINI_TOP_K,
    )
    safety_settings = GEMINI_SAFETY_SETTINGS
    model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME,
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
else:
    logging.error(
        "GOOGLE_API_KEY environment variable not set. Gemini API is required.")
    sys.exit(1)

# --- Initialize Kokoro TTS ---
try:
    tts_pipeline = KPipeline(lang_code='a', repo_id=KOKORO_REPO_ID)
except Exception as e:
    logging.error(f"Error initializing kokoro TTS pipeline: {e}")
    sys.exit(1)

# --- Initialize Whisper STT ---
try:
    whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
    whisper_model = AutoModelForCTC.from_pretrained(WHISPER_MODEL_ID)
    logging.info("OpenAI Whisper STT initialized.")
except Exception as e:
    logging.error(f"Error initializing OpenAI Whisper: {e}")
    sys.exit(1)

# --- Supabase Client ---
SUPABASE_URL = "https://vqukmlutaqthsmaprieb.supabase.co/"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZxdWttbHV0YXF0aHNtYXByaWViIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQ4NzIwMzAsImV4cCI6MjA2MDQ0ODAzMH0.hwwcVLCYDYLVHfajYD77EHT0H2_0GmTv3kAOTLtlpbo"
BUCKET_NAME = "grab-buddy-audio-logs"  # Define Supabase Storage bucket name

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def insert_driver_log(driver_id: str, chat_log: dict) -> dict:
    """
    Inserts a chat log for a driver into the 'driver_logs' table.
    Stores the entire chat history, including both user and assistant messages,
    and now potentially audio URLs.

    Args:
        driver_id (str): Unique identifier of the driver.
        chat_log (dict): The chat log as a dictionary.

    Returns:
        dict: Supabase response.
    """
    try:
        data = {
            "driver_id": driver_id,
            "chat_log": chat_log  # Supabase handles JSON dict automatically
        }
        # ADDED LOGGING
        logging.info(f"Data being inserted into Supabase: {data}")
        response = supabase.table("driver_logs").insert(data).execute()
        return response.data
    except Exception as e:
        return {"error": str(e)}


# --- Audio Playback Functions ---


def play_audio(audio_data, samplerate):
    """Plays audio in a separate thread and handles stop events."""
    global tts_playback_thread
    logging.debug("Starting audio playback thread.")
    tts_playback_thread = threading.Thread(
        target=_play_audio_sync, args=(audio_data, samplerate))
    tts_playback_thread.start()


def _play_audio_sync(audio_data, samplerate):
    """Synchronous audio playback that respects stop events."""
    try:
        samples_generated = 0
        full_audio = np.array([], dtype=np.float32)

        chunk_size = 1024
        for i in range(0, len(audio_data), chunk_size):
            if tts_stop_event.is_set():
                logging.debug("TTS playback interrupted.")
                break
            chunk = audio_data[i:i+chunk_size]
            if chunk.size > 0:
                samples = chunk.shape[0]
                samples_generated += samples
                full_audio = np.concatenate((full_audio, chunk))

        if samples_generated == 0:
            logging.warning(
                "kokoro TTS pipeline did not generate any audio samples.")
        else:
            sd.play(full_audio, samplerate=samplerate)
            while sd.get_stream().active and not tts_stop_event.is_set():
                time.sleep(0.1)
            if tts_stop_event.is_set():
                sd.stop()
                logging.debug("TTS playback stopped.")
    except Exception as e:
        logging.error(f"kokoro TTS Error during playback: {e}")
    finally:
        global tts_playback_thread
        tts_playback_thread = None


# --- Audio Processing and Intent Recognition ---

async def upload_audio_to_supabase(file_path: str, storage_path: str) -> Optional[str]:
    """
    Uploads a single audio file to Supabase Storage and returns the public URL.

    Args:
        file_path (str): Local path to the audio file.
        storage_path (str): Path in Supabase Storage (filename).

    Returns:
        Optional[str]: Public URL of the uploaded file, or None on error.
    """
    try:
        with open(file_path, "rb") as f:
            response = supabase.storage.from_(BUCKET_NAME).upload(
                path=storage_path,
                file=f,
                file_options={"content-type": "audio/wav"},
            )
        # ADDED LOGGING
        logging.info(f"Supabase Storage upload response: {response}")
        if hasattr(response, "path") and response.path:
            file_url = supabase.storage.from_(
                BUCKET_NAME).get_public_url(response.path)
            logging.info(f"Uploaded to Supabase Storage: {storage_path} -> {file_url}")
            return file_url
        else:
            logging.error(
                f"Supabase Storage upload failed for {storage_path}: {response}")
            return None
    except Exception as e:
        logging.error(
            f"Error uploading {storage_path} to Supabase Storage: {e}")
        return None


async def process_audio_chunk(websocket: WebSocket, audio_chunk: np.ndarray, chat_history, driver_id: str):
    """Processes audio chunk: noise reduction, STT, intent, and response, saves and uploads audio."""
    logging.info("Start processing audio chunk")
    samplerate = 16000
    audio_filename = f"{driver_id}_audio_chunk_{int(time.time())}.wav"
    # Save in current directory for simplicity
    audio_filepath = os.path.join(".", audio_filename)

    try:
        # --- SAVE AUDIO CHUNK TO WAV FILE ---
        try:
            sf.write(audio_filepath, audio_chunk, samplerate)
            logging.info(f"Saved audio chunk to: {audio_filepath}")
        except Exception as save_e:
            logging.error(f"Error saving audio file: {save_e}")

        # --- Noise Reduction ---
        reduced_noise = nr.reduce_noise(
            y=audio_chunk,
            sr=samplerate,
            prop_decrease=NOISE_REDUCE_PROP_DECREASE,
            time_constant_s=NOISE_REDUCE_TIME_CONSTANT_S,
        )  # You can tune noise reduction parameters here or disable it
        audio_for_stt = reduced_noise
    except Exception as e:
        logging.warning(
            f"Error applying noise reduction: {e}. Continuing without.")
        audio_for_stt = audio_chunk

    # --- Speech to Text (Whisper) ---
    input_values = whisper_processor(
        audio_for_stt, sampling_rate=samplerate, return_tensors="pt").input_values
    logits = whisper_model(input_values).logits
    predicted_ids = np.argmax(logits.detach().cpu().numpy(), axis=-1)
    transcription = whisper_processor.batch_decode(
        predicted_ids, skip_special_tokens=True)[0]
    query = transcription
    logging.info(f"Whisper Transcription: {query}")

    if not query.strip():
        logging.info("Empty transcription (silence detected in chunk).")
        logging.info("End processing audio chunk (silence)")
        return  # Or handle silence differently if needed

    user_text = query.lower()
    # Add user query to session history, initially without audio_url
    chat_history.append({"role": "user", "content": query, "audio_url": None})
    logging.debug(f"Added user query to chat history: {query}")

    # --- Upload audio to Supabase Storage and get URL ---
    audio_url = await upload_audio_to_supabase(audio_filepath, audio_filename)
    # ADDED LOGGING
    logging.info(f"Audio URL from upload_audio_to_supabase: {audio_url}")
    if audio_url:
        # Add audio_url to chat history
        chat_history[-1]["audio_url"] = audio_url
        logging.debug(f"Added audio_url to chat history: {audio_url}")
        # ADDED LOGGING
        logging.info(f"Chat history after adding audio_url: {chat_history}")

    # --- Intent Recognition (Gemini) ---
    intent_result = await asyncio.to_thread(speech_to_text_with_intent, user_text, chat_history)

    if intent_result:  # Check if intent_result is not None before accessing it
        # Store intent in chat history
        # Add intent to the last user message
        chat_history[-1]["intent"] = intent_result.get("intent", "unknown")
        logging.debug(f"Added intent to chat history: {chat_history[-1]}")

        if intent_result["intent"] != "silence":
            response_text = await asyncio.to_thread(handle_intent_and_get_response, intent_result, chat_history)
            if response_text:
                # Send TTS response over websocket
                # Pass chat_history here
                # await send_tts_response(websocket, response_text, chat_history)
                response_text, state_update = await asyncio.to_thread(handle_intent_and_get_response, intent_result, chat_history)
                await send_tts_response(websocket, response_text, chat_history, state_update)

    # --- Clean up local audio file ---
    try:
        os.remove(audio_filepath)
        logging.info(f"Deleted local audio file: {audio_filepath}")
    except Exception as delete_e:
        logging.warning(
            f"Error deleting local audio file {audio_filepath}: {delete_e}")

    logging.info("End processing audio chunk")
    return intent_result  # return intent result for session management


# Added chat_history parameter
async def send_tts_response(websocket: WebSocket, text_response: str, chat_history, state_update: Optional[dict] = None):
    logging.info(f"Start sending TTS response: '{text_response}'")
    try:
        async with tts_lock:
            llm_text_response = await asyncio.to_thread(generate_tts_with_llm, text_response, chat_history)
            audio_stream = tts_pipeline(text_response, voice=KOKORO_VOICE)

            audio_chunks = []
            for _, _, audio in audio_stream:
                if audio is not None:
                    audio_np = audio.numpy().astype(np.float32) if isinstance(
                        audio, torch.Tensor) else audio.astype(np.float32)
                    audio_chunks.append(audio_np)

            full_audio = np.concatenate(audio_chunks) if audio_chunks else None

            if full_audio is not None:
                with io.BytesIO() as wav_io:
                    sf.write(wav_io, full_audio,
                             KOKORO_SAMPLERATE, format='WAV')
                    wav_io.seek(0)
                    wav_bytes = wav_io.read()
                    await websocket.send_bytes(wav_bytes)

            # Send state update as JSON
            if state_update:
                await websocket.send_text(json.dumps({
                    "type": "state_update",
                    "payload": {
                        "driverStatus": "ONLINE"
                    }
                }))

    except Exception as e:
        logging.error(f"Error during TTS generation or sending: {e}")
    finally:
        if tts_lock.locked():
            tts_lock.release()
        logging.info("End sending TTS response.")



def get_ride_summary_text(limit: int = 3) -> str:
    """
    Fetches ride summaries from Supabase and returns them as a single formatted text string.

    Args:
        limit (int): Number of rides to fetch. Defaults to 3.

    Returns:
        str: Formatted ride summary text.
    """
    try:
        response = supabase.table("rides") \
            .select("pickup_address, dropoff_address, estimated_fare, id") \
            .limit(limit) \
            .execute()

        rides = response.data
        if not rides:
            return "No rides found."

        lines = []
        for idx, ride in enumerate(rides, 1):
            pickup = ride['pickup_address']
            dropoff = ride['dropoff_address']
            fare = float(ride['estimated_fare'])
            lines.append(
                f"{idx} {pickup} to {dropoff}, estimated fare RM {fare:.2f}")

        return ", ".join(lines)

    except Exception as e:
        # Log the error
        logging.error(f"Error fetching ride summary from Supabase: {e}")
        return f"Error fetching ride summary: {e}"


def speech_to_text_with_intent(user_text: str, chat_history) -> Optional[Dict[str, Any]]:
    """Performs intent recognition using Gemini based on user text and chat history."""
    logging.info("Starting intent recognition.")

    intent_prompt_with_history = f"""
    You are an assistant for a Grab driver. You are friendly, concise, and helpful.
    Your primary goal is to determine the user's intent related to ride bookings and general actions, and respond in JSON format.
    Do not use markdown in your response.

    **Chat History:**
    {json.dumps(chat_history)}

    **User Text:** "{user_text}"

    Analyze the user text within the context of the chat history and the provided context prompt.
    Determine the user's intent.  Prioritize these intents if they are clearly present: weather, traffic, directions, and showing bookings. If multiple intents are present, return a 'combined_info' intent with a list of individual intents.

    **Key Intents:**

    1.  **Show Bookings Intent:** User wants to see ride bookings or select a specific ride.
        - **Intent Name:** `"show_bookings"`
        - **Sub-intents and Parameters:**
            - **To show all bookings:** "show my bookings", "my bookings", etc.
            - **To select a specific ride:** "choose number one", "number 2 please", "I want ride 3", destination names.
            - **Parameters:**  `"ride_index"` (integer, 1-indexed) OR `"destination"` (string, ride destination)

    2.  **Choose Booking Intent:** User wants to select a booking.
        - **Intent Name:** `"choose_booking"`
        - **Parameters:** `"ride_index"` (integer, 1-indexed)

    3.  **Confirmation Intent:** User confirms an action.
        - **Intent Name:** `"confirm"`

    4.  **Decline Intent:** User declines an action.
        - **Intent Name:** `"decline"`

    5.  **Directions Intent (General):** User requests directions, optionally with a destination.
        - **Intent Name:** `"directions"`
        - **Parameters:** `"destination"` (string, optional).

    6.  **Show Directions Intent (Ride-Related):** User wants directions for a booked ride.
        - **Intent Name:** `"show_directions"`

    7.  **Traffic Info Intent:** User requests traffic information.
        - **Intent Name:** `"traffic_info"`

    8.  **Weather Info Intent:** User requests weather information.
        - **Intent Name:** `"weather_info"`

    9.  **Emergency Intent:** User indicates an emergency situation.
        - **Intent Name:** `"emergency"`
        - **Examples:** "emergency", "I need help", "accident", "urgent assistance"

    10. **Driver Status Intent:** User wants to go online or offline.
        - **Intent Name:** `"driver_status"`
        - **Parameters:** `"status"` (string: `"online"` or `"offline"`)

    11. **Ride Action Intent:** User wants to accept, decline, start, or end a ride.
        - **Intent Name:** `"ride_action"`
        - **Parameters:** `"action"` (string: `"accept"`, `"decline"`, `"start"`, `"end"`)

    **Combined Intent Handling:**
    If the user expresses multiple intents (e.g., "weather and traffic", "directions and bookings"), return a combined intent:

    **Example for "weather and traffic":**
    ```json
    {{
      "intent": "combined_info",
      "parameters": {{
        "intents": ["weather_info", "traffic_info"]
      }}
    }}
    ```

    **Example for "directions to KLCC and show bookings":**
    ```json
    {{
      "intent": "combined_info",
      "parameters": {{
        "intents": ["directions", "show_bookings"],
        "directions_parameters": {{ "destination": "KLCC" }}
      }}
    }}
    ```

    If only one intent is clearly identified, return that single intent as before. If intent is unclear, return `"intent": "unknown"`.

    JSON Response:
    """  # Intent definitions - refer to standalone code

    logging.debug(f"Intent Prompt:\n{intent_prompt_with_history}")
    try:
        response = model.generate_content(intent_prompt_with_history)
        json_string_raw = response.text
        json_string = json_string_raw.replace(
            "```json", "").replace("```", "").strip()
        logging.debug(f"Stripped JSON String: {json_string}")
        logging.info(f"LLM Intent JSON (Raw): {json_string_raw}")
        intent_json = json.loads(json_string)
        logging.info(
            f"Detected intent: {intent_json.get('intent', 'unknown')}")
        return intent_json
    except json.JSONDecodeError as e:
        logging.error(f"JSON Decode Error: {e}, Raw response: {response.text}")
        return {"intent": "unknown"}
    except Exception as e:
        logging.error(f"LLM Intent Error: {e}")
        return {"intent": "unknown"}


def generate_tts_with_llm(text_prompt: str, chat_history) -> str:
    """Generates TTS prompt using LLM based on text prompt and chat history."""
    logging.debug(f"Generating TTS prompt for: {text_prompt}")
    # Log history before
    logging.debug(f"Chat History before LLM generation: {chat_history}")
    prompt_with_history = f"""
        You are a helpful and concise assistant for a Grab driver.
        Generate a short, natural-sounding spoken response for the driver based on the following:

        **Chat History:**
        {json.dumps(chat_history)}

        **Driver Request:**
        {text_prompt}

        Do not use markdown. Just provide the plain text response designed to be spoken aloud. Keep it very short and direct.
        """
    try:
        response = model.generate_content(prompt_with_history)
        llm_generated_text = response.text.strip()
        if llm_generated_text:
            logging.info(f"LLM generated TTS prompt: {llm_generated_text}")
            # Add assistant response to history - SYSTEM CHAT IS STORED HERE
            chat_history.append(
                {"role": "assistant", "content": llm_generated_text})
            # Log history after
            logging.debug(f"Chat History after LLM generation: {chat_history}")
            logging.debug(
                f"Added assistant response to chat history: {llm_generated_text}")
            return llm_generated_text
        else:
            logging.warning("LLM did not generate text for TTS.")
            return "Sorry, I encountered an issue."
    except Exception as e:
        logging.error(f"Error generating TTS prompt with LLM: {e}")
        return "Sorry, I encountered an issue."


def show_directions(destination="Universiti Malaya Faculty of Computer Science"):
    tts_text = f"Showing directions to {destination}."
    text_to_speech(tts_text)
    logging.info(f"Showing directions to {destination} (Placeholder).")


async def ask_user_to_show_bookings(ride_list: List[str], chat_history) -> Optional[str or Dict[str, Any]]:
    # ... (Implementation of ask_user_to_show_bookings - same logic as standalone, adjust for async if needed) ...
    pass  # Replace with actual implementation from your standalone code, adjusted for async


async def read_ride_details_and_confirm(ride_details: str, chat_history) -> bool:
    # ... (Implementation of read_ride_details_and_confirm - same logic as standalone, adjust for async) ...
    pass  # Replace with actual implementation from your standalone code, adjusted for async


async def ask_to_show_directions_and_confirm(chat_history) -> bool:
    # ... (Implementation of ask_to_show_directions_and_confirm - same logic as standalone, adjust for async) ...
    pass  # Replace with actual implementation from your standalone code, adjusted for async


def get_traffic_info():
    return "Traffic is currently light."


def get_weather_info():
    return "The weather is sunny and 25 degrees Celsius."


def handle_multiple_intents_tts(intents):
    """Handles multiple intents and generates TTS responses directly."""
    logging.info(f"Handling combined intents for TTS: {intents}")
    response_parts = []
    for intent in intents:
        if intent == "weather_info":
            response_parts.append(get_weather_info())
        elif intent == "traffic_info":
            response_parts.append(get_traffic_info())
        elif intent == "directions":
            response_parts.append("Showing directions.")
        elif intent == "show_bookings":
            response_parts.append("Here are your bookings.")
        else:
            response_parts.append(f"Handling intent: {intent}")

    combined_response = " ".join(response_parts)
    text_to_speech(combined_response)


def get_ride_details_from_text(ride_text: str) -> str:
    return "Ride details not available."


async def book_ride(ride_index: int) -> str:
    """Simulates booking a ride.  Replace with actual Supabase logic."""
    try:
        response = supabase.table("rides") \
            .select("pickup_address, dropoff_address, estimated_fare, id") \
            .limit(ride_index) \
            .execute()
        rides = response.data
        chosen_ride = rides[ride_index - 1]
        pickup = chosen_ride['pickup_address']
        dropoff = chosen_ride['dropoff_address']
        fare = chosen_ride['estimated_fare']

        # Simulate updating the booking status in the database
        # In a real application, you'd update a "status" field to "booked" or similar

        # Generate a confirmation message
        confirmation_message = (
            f"Booking confirmed!  Ride from {pickup} to {dropoff}, "
            f"estimated fare RM {fare:.2f}.  The passenger has been notified."
        )
        return confirmation_message

    except Exception as e:
        logging.error(f"Error booking ride: {e}")
        return "Sorry, there was an error booking the ride."


def handle_intent_and_get_response(intent_result, chat_history):
    """Handles intent and returns a text response (used in WebSocket context)."""
    logging.debug(f"Handling intent: {intent_result}")
    response_parts = []
    state_update = {}

    if intent_result["intent"] == "combined_info":
        if "parameters" in intent_result and "intents" in intent_result["parameters"]:
            intents = intent_result["parameters"]["intents"]
            if "weather_info" in intents:
                response_parts.append(get_weather_info())
            if "traffic_info" in intents:
                response_parts.append(get_traffic_info())
            if "emergency" in intents:
                response_parts.append(
                    "Grab is notified of this issue and it is being looked at.")
            if "directions" in intents:
                destination_for_directions = intent_result["parameters"].get(
                    "directions_parameters", {}).get("destination")
                if not destination_for_directions:
                    destination_for_directions = last_chosen_destination_global if last_chosen_destination_global else "Universiti Malaya Faculty of Computer Science"  # Use global
                response_parts.append(
                    f"Showing directions to {destination_for_directions}.")
            if "show_bookings" in intents:
                # Fetch ride summaries from Supabase
                ride_summaries = get_ride_summary_text(limit=3)
                response_parts.append("Here are your bookings:")
                response_parts.append(ride_summaries)  # Use fetched data
                # response_parts.extend(sample_rides)

    elif intent_result["intent"] == "weather_info":
        response_parts.append(get_weather_info())
    elif intent_result["intent"] == "traffic_info":
        response_parts.append(get_traffic_info())
    elif intent_result["intent"] == "directions":
        destination_for_directions = intent_result.get(
            "parameters", {}).get("destination")
        if not destination_for_directions:
            destination_for_directions = last_chosen_destination_global if last_chosen_destination_global else "Universiti Malaya Faculty of Computer Science"  # Use global
        response_parts.append(
            f"Showing directions to {destination_for_directions}.")
    elif intent_result["intent"] == "show_bookings":
        # Fetch ride summaries from Supabase
        ride_summaries = get_ride_summary_text(limit=3)
        response_parts.append("Here are your bookings:")
        response_parts.append(ride_summaries)  # Use fetched data
        # response_parts.extend(sample_rides)

    elif intent_result["intent"] == "choose_booking":
        ride_index = intent_result.get("parameters", {}).get("ride_index")
        if ride_index is not None:
            ride_index = int(ride_index)
            # Attempt to book the ride
            # Book ride asynchronously
            booking_result = asyncio.run(book_ride(ride_index))
            response_parts.append(booking_result)  # Add booking confirmation
        else:
            response_parts.append(
                "Sorry, I couldn't determine which ride to book.")

    elif intent_result["intent"] == "unknown":
        response_parts.append(
            "")  # Or a default "I didn't understand" message
    elif intent_result["intent"] == "error":
        response_parts.append("Sorry, there was an error. Please try again.")
    elif intent_result["intent"] == "silence":
        response_parts.append("I didn't hear anything. Please speak again.")
    elif intent_result["intent"] == "emergency":  # Added emergency intent
        response_parts.append(
            "Grab is notified of this issue and it is being looked at.")
    elif intent_result["intent"] == "driver_status":
        status = intent_result.get("parameters", {}).get("status")
        if status == "online":
            response_parts.append("You are now online.")
            state_update = {"driverStatus": "ONLINE"}
        elif status == "offline":
            response_parts.append("You are now offline.")
            state_update = {"driverStatus": "OFFLINE"}

    elif intent_result["intent"] == "ride_action":
        action = intent_result.get("parameters", {}).get("action")
        if action == "accept":
            response_parts.append("Ride accepted.")
            state_update = {"rideStatus": "ACCEPTED"}
        elif action == "decline":
            response_parts.append("Ride declined.")
            state_update = {"rideStatus": "DECLINED"}
        elif action == "start":
            response_parts.append("Ride started.")
            state_update = {"rideStatus": "STARTED"}
        elif action == "end":
            response_parts.append("Ride ended.")
            state_update = {"rideStatus": "ENDED"}

    return "\n".join(response_parts), state_update

# --- WebSocket Endpoint ---


@app.websocket("/ws/grab_buddy")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection accepted.")
    # Session-specific chat history, stores both user and assistant messages
    chat_history_session = []
    consecutive_unknown_intents_session = 0
    # Generate unique driver ID for each session
    driver_id = "c00a8d24-e297-4f12-9b8e-5d78cf28b5d0"
    logging.info(f"Driver ID for this session: {driver_id}")

    try:
        # Fixed initial greeting text - as requested by user
        initial_greeting_text = "Hello this is Grab Buddy! How can I help you?"
        # Send greeting via TTS
        # Pass history here
        await send_tts_response(websocket, initial_greeting_text, chat_history_session)

        while True:
            data = await websocket.receive_bytes()  # Receive audio data from WebSocket
            # Convert bytes to numpy float32 array
            audio_np = np.frombuffer(data, dtype=np.float32)

            # Process audio chunk, now passing driver_id
            intent_result = await process_audio_chunk(websocket, audio_np, chat_history_session, driver_id)

            if intent_result and intent_result["intent"] == "unknown":
                # Fallback for unknown intent in WebSocket context
                # Send simple fallback TTS
                # Pass history here
                await send_tts_response(websocket, "", chat_history_session)

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected.")
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        logging.info("WebSocket connection closed.")
        # Log chat history content for verification
        logging.debug(
            f"Chat history being logged to Supabase: {chat_history_session}")
        # Log chat history to Supabase after connection is closed
        log_result = insert_driver_log(
            driver_id, {"messages": chat_history_session})
        logging.info(f"Supabase log result on connection close: {log_result}")


if __name__ == '__main__':
    import uvicorn
    warnings.filterwarnings(
        "ignore",
        message="dropout option adds dropout after all but last recurrent layer.*",
        module="torch.nn.modules.rnn"
    )
    warnings.filterwarnings(
        "ignore",
        message="`torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.*",
        module="torch.nn.utils.weight_norm"
    )
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
