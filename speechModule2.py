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

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
NOISE_REDUCE_PROP_DECREASE = 0.9
NOISE_REDUCE_TIME_CONSTANT_S = 0.5
SPEECH_RECOGNITION_TIMEOUT = 10  # Increased speech recognition timeout
TTS_DELAY = 0.5  # Delay after TTS before listening again
KOKORO_REPO_ID = 'hexgrad/Kokoro-82M'  # Define Kokoro repo_id
# Maximum consecutive unknown intents before fallback
MAX_CONSECUTIVE_UNKNOWN_INTENTS = 3

USE_GEMINI = bool(GOOGLE_API_KEY)
tts_pipeline = None
whisper_processor = None
whisper_model = None
chat_history = []
tts_stop_event = threading.Event()  # Event to stop TTS
tts_playback_thread = None  # Thread for TTS playback

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

try:
    # Explicitly set repo_id
    tts_pipeline = KPipeline(lang_code='a', repo_id=KOKORO_REPO_ID)
except Exception as e:
    logging.error(f"Error initializing kokoro TTS pipeline: {e}")
    sys.exit(1)

try:
    whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
    whisper_model = AutoModelForCTC.from_pretrained(WHISPER_MODEL_ID)
    logging.info("OpenAI Whisper STT initialized.")
except Exception as e:
    logging.error(f"Error initializing OpenAI Whisper: {e}")
    sys.exit(1)


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

        # Process audio in chunks to check stop event more frequently if needed for very long TTS
        chunk_size = 1024  # Adjust chunk size as needed
        for i in range(0, len(audio_data), chunk_size):
            if tts_stop_event.is_set():
                logging.debug("TTS playback interrupted.")
                break
            chunk = audio_data[i:i+chunk_size]
            if chunk.size > 0:  # Handle potentially empty last chunk
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
        tts_playback_thread = None  # Reset thread variable after playback


def text_to_speech(text: str):
    """Converts text to speech using the kokoro TTS pipeline and waits for playback to finish."""
    logging.debug(f"Text to speech: {text}")
    tts_stop_event.clear()  # Reset stop event
    try:
        full_audio = np.array([], dtype=np.float32)
        for _, _, audio in tts_pipeline(text, voice=KOKORO_VOICE):
            if tts_stop_event.is_set():  # Check stop event within loop
                logging.debug("TTS playback interrupted.")
                break
            if audio is not None:
                # Convert torch.Tensor to numpy array if needed
                if isinstance(audio, torch.Tensor):
                    audio_np = audio.numpy()
                else:
                    audio_np = audio

                # Debugging to check audio type and value
                logging.debug(f"Type of audio: {type(audio)}")
                if isinstance(audio, np.ndarray):
                    logging.debug(f"Audio shape: {audio.shape}")
                elif isinstance(audio, torch.Tensor):
                    logging.debug(f"Audio shape (torch): {audio.shape}")
                else:
                    # Print if not numpy array or tensor
                    logging.debug(f"Audio value: {audio}")

                # Ensure we are working with numpy array for shape and concat
                if isinstance(audio_np, np.ndarray):
                    samples = audio_np.shape[0]
                    if samples > 0:
                        full_audio = np.concatenate((full_audio, audio_np))
                else:
                    logging.warning(
                        "Audio data is not a numpy array after conversion. Skipping this chunk.")

        if full_audio.size > 0:
            play_audio(full_audio, KOKORO_SAMPLERATE)
            if tts_playback_thread and tts_playback_thread.is_alive():  # Wait for playback to finish
                tts_playback_thread.join()
                logging.debug("TTS playback finished and thread joined.")
        else:
            logging.warning("No audio data generated by TTS.")

    except Exception as e:
        logging.error(f"kokoro TTS Error: {e}")


def speech_to_text_with_intent(context_prompt: str = "") -> Optional[Dict[str, Any]]:
    """Records audio, performs speech-to-text, and determines intent using Gemini."""
    logging.info("Starting speech-to-text and intent recognition.")
    samplerate = 16000
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=samplerate) as source:
        logging.info("Listening...")
        try:
            time.sleep(TTS_DELAY)  # Add delay before listening
            audio_data = r.listen(
                source, phrase_time_limit=SPEECH_RECOGNITION_TIMEOUT)
            tts_stop_event.set()  # Stop TTS immediately after listening starts to avoid overlap
        except sr.WaitTimeoutError:
            logging.warning("No speech detected within timeout.")
            return {"intent": "silence"}

    audio_np = np.frombuffer(audio_data.get_raw_data(
        convert_rate=samplerate, convert_width=2), dtype=np.int16)
    audio_float32 = audio_np.astype(np.float32) / 32767.0

    try:
        reduced_noise = nr.reduce_noise(
            y=audio_float32,
            sr=samplerate,
            prop_decrease=NOISE_REDUCE_PROP_DECREASE,
            time_constant_s=NOISE_REDUCE_TIME_CONSTANT_S,
        )
        logging.debug("Noise reduction applied.")
        audio_for_stt = reduced_noise
    except Exception as e:
        logging.warning(
            f"Error applying noise reduction: {e}. Continuing without noise reduction.")
        audio_for_stt = audio_float32

    logging.debug("Using OpenAI Whisper for Speech Recognition...")
    input_values = whisper_processor(
        audio_for_stt, sampling_rate=samplerate, return_tensors="pt").input_values
    logits = whisper_model(input_values).logits
    predicted_ids = np.argmax(logits.detach().cpu().numpy(), axis=-1)
    transcription = whisper_processor.batch_decode(
        predicted_ids, skip_special_tokens=True)[0]
    query = transcription
    logging.info(f"Whisper Transcription: {query}")

    if not query.strip():
        logging.info("Empty transcription (silence detected).")
        return {"intent": "silence"}

    user_text = query.lower()
    chat_history.append({"role": "user", "content": query}
                        )  # Add user query to chat history
    logging.debug(f"Added user query to chat history: {query}")

    intent_prompt_with_history = f"""
    You are an assistant for a Grab driver. You are friendly, concise, and helpful.
    Your primary goal is to determine the user's intent related to ride bookings and general actions, and respond in JSON format.
    Do not use markdown in your response.

    **Chat History:**
    {json.dumps(chat_history)}

    **Context Prompt:**
    {context_prompt}

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

    2.  **Confirmation Intent:** User confirms an action.
        - **Intent Name:** `"confirm"`

    3.  **Decline Intent:** User declines an action.
        - **Intent Name:** `"decline"`

    4.  **Directions Intent (General):** User requests directions, optionally with a destination.
        - **Intent Name:** `"directions"`
        - **Parameters:** `"destination"` (string, optional).

    5.  **Show Directions Intent (Ride-Related):** User wants directions for a booked ride.
        - **Intent Name:** `"show_directions"`

    6.  **Traffic Info Intent:** User requests traffic information.
        - **Intent Name:** `"traffic_info"`

    7.  **Weather Info Intent:** User requests weather information.
        - **Intent Name:** `"weather_info"`

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
    """
    logging.debug(
        f"Intent Prompt:\n{intent_prompt_with_history}")  # Log intent prompt
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


def generate_tts_with_llm(text_prompt: str) -> str:
    """Generates TTS prompt using LLM. Consider simplifying or removing if possible."""
    logging.debug(f"Generating TTS prompt for: {text_prompt}")
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
            chat_history.append(
                # Keep history if needed for context
                {"role": "assistant", "content": llm_generated_text})
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
    tts_text = f"Showing directions to {destination}."  # Direct TTS text
    text_to_speech(tts_text)
    logging.info(f"Showing directions to {destination} (Placeholder).")


def ask_user_to_show_bookings(ride_list: List[str]) -> Optional[str or Dict[str, Any]]:
    logging.info("Asking user to show bookings.")
    if not ride_list:
        tts_text = "No rides available."  # Direct TTS text
        text_to_speech(tts_text)
        return None

    ride_options_text = "\n".join(
        [f"{i+1}. {ride}" for i, ride in enumerate(ride_list)])
    # Base TTS prompt
    # tts_prompt_base = f"Here are your bookings: {ride_options_text}. "
    # Use LLM for final phrasing but keep it concise
    text_to_speech("Which ride would you like to take?")

    retries = 3
    chosen_ride = None
    destination_to_ride_index = {}
    for i, ride in enumerate(ride_list):
        try:
            if ride.startswith("Ride to "):
                ride_no_prefix = ride[len("Ride to "):]
                parts_from = ride_no_prefix.split(" from ")
                if len(parts_from) > 1:
                    destination = parts_from[0].strip()
                else:
                    destination = ride_no_prefix.split(",")[0].strip()
                destination_to_ride_index[destination.lower()] = i + 1
        except IndexError as e:
            logging.error(f"Error parsing ride string: {ride}, Error: {e}")
            continue

    while retries >= 0 and chosen_ride is None:
        # text_to_speech(tts_prompt_full)
        context_for_intent = f"User is choosing a ride from list: {ride_options_text}. Can say 'number one', 'number two', etc., or destination name."
        intent_result = speech_to_text_with_intent(
            context_prompt=context_for_intent)
        logging.info(f"Intent Result: {intent_result}")

        if intent_result and intent_result["intent"] == "show_bookings":
            if "parameters" in intent_result and "ride_index" in intent_result["parameters"]:
                try:
                    ride_index = intent_result["parameters"].get("ride_index")
                    if ride_index is not None and 1 <= ride_index <= len(ride_list):
                        chosen_ride = ride_list[ride_index - 1]
                        logging.info(f"User chose ride number: {chosen_ride}")
                        return chosen_ride
                    else:
                        logging.warning("Invalid ride index chosen.")
                        tts_text_invalid_ride = "Invalid ride number. Please choose from the list."  # Direct TTS
                        text_to_speech(tts_text_invalid_ride)
                except (KeyError, TypeError):
                    logging.error("Error parsing ride index from intent.")
                    tts_text_choose_again = "Please choose the ride number again."  # Direct TTS
                    text_to_speech(tts_text_choose_again)
            elif "parameters" in intent_result and "destination" in intent_result["parameters"]:
                chosen_destination = intent_result["parameters"].get(
                    "destination")
                ride_index_by_destination = destination_to_ride_index.get(
                    chosen_destination.lower())
                if ride_index_by_destination:
                    chosen_ride = ride_list[ride_index_by_destination - 1]
                    logging.info(
                        f"User chose ride by destination: {chosen_ride}")
                    return chosen_ride
                else:
                    logging.warning(
                        f"No ride found for destination: {chosen_destination}")
                    tts_text_no_ride_dest = "Sorry, I didn't find a ride for that destination. Please choose by number."  # Direct TTS
                    text_to_speech(tts_text_no_ride_dest)
            else:
                tts_text_choose_list = "Please choose a ride number or destination from the list."  # Direct TTS
                text_to_speech(tts_text_choose_list)
                tts_prompt_full = generate_tts_with_llm(
                    "Please choose a ride number or destination.")  # Re-generate prompt if needed

        elif intent_result and intent_result["intent"] == "unknown":
            if "raw_text" in intent_result:
                raw_text = intent_result["raw_text"].lower()
                logging.info(f"Unknown intent, raw user text: {raw_text}")
                for dest_lower, ride_index in destination_to_ride_index.items():
                    if dest_lower == raw_text.strip():
                        chosen_ride = ride_list[ride_index - 1]
                        logging.info(
                            f"Interpreted unknown intent as destination selection: {chosen_ride}")
                        return chosen_ride
            tts_text_unknown = "Please specify the ride number or destination."  # Direct TTS
            text_to_speech(tts_text_unknown)

        elif intent_result and intent_result["intent"] == "error":
            pass
        elif intent_result and intent_result["intent"] == "silence":
            return None
        elif intent_result and intent_result["intent"] in ["confirm", "decline"]:
            logging.warning(
                f"Ignoring unexpected intent in ride selection: {intent_result['intent']}")
            tts_prompt_full = generate_tts_with_llm(
                # Re-generate if unexpected intent
                "Please choose a ride number or destination from the list.")

        elif intent_result and intent_result["intent"] == "combined_info":
            if "parameters" in intent_result and "intents" in intent_result["parameters"]:
                # Handle combined intents directly
                handle_multiple_intents_tts(
                    intent_result["parameters"]["intents"])
        elif intent_result and intent_result["intent"] not in ["show_bookings", "unknown", "error", "silence", "confirm", "decline", "combined_info"]:
            logging.warning(
                f"Unexpected intent during ride selection: {intent_result['intent']}")
            return intent_result
        else:
            logging.warning(
                f"Unexpected intent result during ride selection: {intent_result}")
            tts_text_unexpected_result = "Please choose a ride number or destination from the list."  # Direct TTS
            text_to_speech(tts_text_unexpected_result)
            tts_prompt_full = generate_tts_with_llm(
                "Please choose a ride number or destination.")  # Re-generate prompt if needed

        retries -= 1
        if retries >= 0 and chosen_ride is None:
            if intent_result and intent_result["intent"] != "show_bookings":
                tts_prompt_full = generate_tts_with_llm(
                    "Please choose a ride from the list again.")  # Re-generate prompt for retry

    if chosen_ride:
        return chosen_ride
    else:
        tts_text_retry_fail = "Sorry, please try again."  # Direct TTS
        text_to_speech(tts_text_retry_fail)
        return None


def read_ride_details_and_confirm(ride_details: str) -> bool:
    logging.info("Reading ride details and asking for confirmation.")
    details_text = f"Ride details: {ride_details}."
    # Use LLM for phrasing confirmation prompt
    tts_prompt_full = generate_tts_with_llm(
        f"Confirm these ride details: {details_text}. Do you want to confirm or ask about weather or traffic?")

    retries = 2
    confirmed = False
    while retries >= 0 and not confirmed:
        text_to_speech(tts_prompt_full)
        confirmation_intent = speech_to_text_with_intent(
            context_prompt="Confirm ride details. User can say yes/no or ask about weather/traffic.")

        if confirmation_intent and confirmation_intent["intent"] == "confirm":
            confirmed = True
            break
        elif confirmation_intent and confirmation_intent["intent"] == "decline":
            tts_text_ride_decline = "Ride declined."  # Direct TTS
            text_to_speech(tts_text_ride_decline)
            return False
        elif confirmation_intent and confirmation_intent["intent"] == "silence":
            pass
        elif confirmation_intent and confirmation_intent["intent"] == "unknown":
            tts_text_confirm_yes_no = "Please say yes/no to confirm, or ask weather/traffic."  # Direct TTS
            text_to_speech(tts_text_confirm_yes_no)
        elif confirmation_intent and confirmation_intent["intent"] == "error":
            pass
        elif confirmation_intent and confirmation_intent["intent"] == "weather_info":
            weather_info_text = get_weather_info()
            text_to_speech(weather_info_text)  # Directly TTS info
        elif confirmation_intent and confirmation_intent["intent"] == "traffic_info":
            traffic_info_text = get_traffic_info()
            text_to_speech(traffic_info_text)  # Directly TTS info
        elif confirmation_intent and confirmation_intent["intent"] == "combined_info":
            if "parameters" in confirmation_intent and "intents" in confirmation_intent["parameters"]:
                # Handle combined intents directly
                handle_multiple_intents_tts(
                    confirmation_intent["parameters"]["intents"])
        else:
            tts_text_confirm_yes_no_again = "Please say yes/no to confirm, or ask weather/traffic."  # Direct TTS
            text_to_speech(tts_text_confirm_yes_no_again)

        retries -= 1
        if retries >= 0 and not confirmed:
            tts_prompt_full = generate_tts_with_llm(
                "Confirm again: yes/no, or ask weather/traffic.")  # Re-generate prompt for retry
        elif retries < 0 and not confirmed:
            tts_text_try_later = "Sorry, please try again later."  # Direct TTS
            text_to_speech(tts_text_try_later)
            return False
    return confirmed


def ask_to_show_directions_and_confirm() -> bool:
    logging.info("Asking user if they want directions.")
    tts_prompt_full = "Do you want to see directions for your ride?" # Initialize tts_prompt_full to prevent UnboundLocalError
    text_to_speech(tts_prompt_full)

    retries = 2
    show_directions_confirmed = False
    while retries >= 0 and not show_directions_confirmed:
        text_to_speech(tts_prompt_full)
        directions_intent = speech_to_text_with_intent(
            context_prompt="Ask user if they want to show directions (yes/no).")

        if directions_intent and directions_intent["intent"] == "show_directions":
            show_directions_confirmed = True
            break
        elif directions_intent and directions_intent["intent"] == "decline":
            tts_text_directions_decline = "Directions declined."  # Direct TTS
            text_to_speech(tts_text_directions_decline)
            return False
        elif directions_intent and directions_intent["intent"] == "silence":
            pass
        elif directions_intent and directions_intent["intent"] == "unknown":
            tts_text_directions_yn_please = "Do you want to see directions? Please say yes/no."  # Direct TTS
            text_to_speech(tts_text_directions_yn_please)
            tts_prompt_full = tts_text_directions_yn_please # Update for retry
        elif directions_intent and directions_intent["intent"] == "error":
            pass
        else:
            tts_text_directions_yn_again = "Do you want to see directions? Please say yes/no."  # Direct TTS
            text_to_speech(tts_text_directions_yn_again)
            tts_prompt_full = tts_text_directions_yn_again # Update for retry

        retries -= 1
        if retries >= 0 and not show_directions_confirmed:
            tts_prompt_full = generate_tts_with_llm(
                "Again: Do you want to see directions?")  # Re-generate prompt for retry
        elif retries < 0 and not show_directions_confirmed:
            tts_text_try_later_directions = "Sorry, please try again later."  # Direct TTS
            text_to_speech(tts_text_try_later_directions)
            return False
    return show_directions_confirmed


def get_traffic_info():
    return "Traffic is currently light."


def get_weather_info():
    return "The weather is sunny and 25 degrees Celsius."


def handle_multiple_intents_tts(intents):
    """Handles multiple intents and generates TTS responses directly."""
    logging.info(f"Handling combined intents: {intents}")
    response_parts = []
    for intent in intents:
        if intent == "weather_info":
            response_parts.append(get_weather_info())
        elif intent == "traffic_info":
            response_parts.append(get_traffic_info())
        elif intent == "directions":
            # Assuming directions intent is handled separately if destination needed
            # Basic response, enhance as needed
            response_parts.append("Showing directions.")
        elif intent == "show_bookings":
            # Basic response, bookings list handled elsewhere
            response_parts.append("Here are your bookings.")
        else:
            # Fallback for other intents
            response_parts.append(f"Handling intent: {intent}")

    combined_response = " ".join(response_parts)
    text_to_speech(combined_response)


def get_ride_details_from_text(ride_text: str) -> str:
    return "Ride details not available."


def handle_multiple_intents(intent_result):
    """Handles multiple intents for textual responses (used in main loop before TTS)."""
    logging.debug(f"Handling multiple intents: {intent_result}")
    response_parts = []
    if intent_result["intent"] == "combined_info":
        if "parameters" in intent_result and "intents" in intent_result["parameters"]:
            intents = intent_result["parameters"]["intents"]
            if "weather_info" in intents:
                response_parts.append(get_weather_info())
            if "traffic_info" in intents:
                response_parts.append(get_traffic_info())
            if "directions" in intents:  # Handle directions in combined intent response
                destination_for_directions = intent_result["parameters"].get(
                    "directions_parameters", {}).get("destination")
                if not destination_for_directions:
                    destination_for_directions = last_chosen_destination if last_chosen_destination else "Universiti Malaya Faculty of Computer Science"
                response_parts.append(
                    f"Showing directions to {destination_for_directions}.")
            if "show_bookings" in intents:  # Handle show bookings in combined intent response
                response_parts.append("Here are your bookings:")
                # Add sample rides or fetch real bookings here
                response_parts.extend(sample_rides)

    elif intent_result["intent"] == "weather_info":
        response_parts.append(get_weather_info())
    elif intent_result["intent"] == "traffic_info":
        response_parts.append(get_traffic_info())
    elif intent_result["intent"] == "directions":
        destination_for_directions = intent_result.get(
            "parameters", {}).get("destination")
        if not destination_for_directions:
            destination_for_directions = last_chosen_destination if last_chosen_destination else "Universiti Malaya Faculty of Computer Science"
        response_parts.append(
            f"Showing directions to {destination_for_directions}.")
    elif intent_result["intent"] == "show_bookings":
        response_parts.append("Here are your bookings:")
        # Add sample rides or fetch real bookings here
        response_parts.extend(sample_rides)
    elif intent_result["intent"] == "unknown":
        response_parts.append(
            # Modified unknown response
            "I'm still learning, could you please rephrase your request?")
    elif intent_result["intent"] == "error":
        response_parts.append("Sorry, there was an error. Please try again.")
    elif intent_result["intent"] == "silence":
        response_parts.append("I didn't hear anything. Please speak again.")

    return "\n".join(response_parts)


if __name__ == '__main__':
    # Suppress specific warnings from Kokoro TTS - RNN dropout and weight_norm (optional)
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

    try:
        # Initialize KPipeline with explicit repo_id
        KPipeline(lang_code='a', repo_id=KOKORO_REPO_ID)
        initial_greeting_text = "Hello this is Grab Buddy! How can I help you?"  # Direct TTS text
        text_to_speech(initial_greeting_text)
    except RuntimeError:
        logging.error("Kokoro TTS Pipeline likely not initialized in test.")

    logging.info("Grab Driver Buddy Ready! (Ctrl+C to exit)")
    sample_rides = ["Ride to Sunway Universiti from Universiti Tower, RM10",
                    "Ride to KLIA from KLCC, RM60", "Ride to Monash from Sunway Universiti, RM35"]

    last_chosen_destination = None
    ride_confirmed = False
    # Initialize counter for consecutive unknown intents
    consecutive_unknown_intents = 0

    while True:
        chosen_ride_text = None
        unexpected_intent_result = None
        intent_result_main = speech_to_text_with_intent()
        logging.info(f"Main Loop Intent Result: {intent_result_main}")

        if intent_result_main:
            # Handle combined intents directly for TTS in main loop
            if intent_result_main["intent"] == "combined_info":
                if "parameters" in intent_result_main and "intents" in intent_result_main["parameters"]:
                    handle_multiple_intents_tts(
                        intent_result_main["parameters"]["intents"])
                consecutive_unknown_intents = 0  # Reset counter on valid intent
            elif intent_result_main["intent"] == "unknown":
                consecutive_unknown_intents += 1  # Increment counter for unknown intent
                if consecutive_unknown_intents >= MAX_CONSECUTIVE_UNKNOWN_INTENTS:
                    # Different fallback response
                    tts_prompt = "Sorry, I'm having trouble understanding. Let's start over. How can I help you?"
                    text_to_speech(tts_prompt)
                    chat_history = []  # Optionally clear chat history to reset context
                    consecutive_unknown_intents = 0  # Reset counter after fallback
                    logging.info(
                        "Max consecutive unknown intents reached. Resetting.")
                    continue  # Skip further processing and restart loop
                else:
                    # Use handle_multiple_intents for "unknown"
                    combined_response = handle_multiple_intents(
                        intent_result_main)
                    # No need for LLM prompt generation for simple responses
                    tts_prompt = combined_response
                    text_to_speech(tts_prompt)
            else:
                combined_response = handle_multiple_intents(
                    intent_result_main)  # Corrected function call
                # No need for LLM prompt generation for simple responses
                tts_prompt = combined_response
                text_to_speech(tts_prompt)
                consecutive_unknown_intents = 0  # Reset counter on valid intent

            if intent_result_main["intent"] in ["weather_info", "traffic_info", "combined_info", "directions", "unknown", "error", "silence"]:
                continue

        ride_confirmed = False
        while chosen_ride_text is None and unexpected_intent_result is None:
            booking_result = ask_user_to_show_bookings(sample_rides)
            if isinstance(booking_result, dict):
                unexpected_intent_result = booking_result
                break
            else:
                chosen_ride_text = booking_result

            if chosen_ride_text:
                logging.info(f"User's ride choice text: {chosen_ride_text}")
                last_chosen_destination = chosen_ride_text.split(" to ")[
                    1].split(",")[0]
                break
            else:
                logging.info("No ride chosen, returning to main loop.")
                continue

        if unexpected_intent_result:
            combined_response = handle_multiple_intents(
                unexpected_intent_result)
            tts_prompt = combined_response  # No LLM TTS prompt needed for simple responses
            text_to_speech(tts_prompt)
            continue

        if not chosen_ride_text:
            continue

        sample_ride_details = get_ride_details_from_text(chosen_ride_text)
        ride_confirmed = read_ride_details_and_confirm(sample_ride_details)
        logging.info(f"Ride confirmed: {ride_confirmed}")

        if not ride_confirmed:
            continue

        show_directions_for_ride = ask_to_show_directions_and_confirm()
        logging.info(
            f"Show directions (ride related): {show_directions_for_ride}")

        final_confirmation_text = "Ride booked. Thank you!"  # Direct TTS text
        text_to_speech(final_confirmation_text)
        logging.info(
            "Ride booking flow completed. Listening for next command.\n")
        chat_history = []
        consecutive_unknown_intents = 0  # Reset counter after completing a flow