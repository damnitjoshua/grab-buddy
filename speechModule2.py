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

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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

USE_GEMINI = bool(GOOGLE_API_KEY)
tts_pipeline = None
whisper_processor = None
whisper_model = None
chat_history = []

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
    print("Error: GOOGLE_API_KEY environment variable not set. Gemini API is required for this version.")
    sys.exit(1)

try:
    tts_pipeline = KPipeline(lang_code='a')
except Exception as e:
    print(f"Error initializing kokoro TTS pipeline: {e}")
    sys.exit(1)

try:
    whisper_processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
    whisper_model = AutoModelForCTC.from_pretrained(WHISPER_MODEL_ID)
    print("OpenAI Whisper STT initialized.")
except Exception as e:
    print(f"Error initializing OpenAI Whisper: {e}")
    sys.exit(1)


def text_to_speech(text: str):
    try:
        samples_generated = 0
        full_audio = np.array([], dtype=np.float32)
        for _, _, audio in tts_pipeline(text, voice=KOKORO_VOICE):
            if audio is not None:
                samples = audio.shape[0]
                if samples > 0:
                    samples_generated += samples
                    full_audio = np.concatenate((full_audio, audio))
        if samples_generated == 0:
            print("Error: kokoro TTS pipeline did not generate any audio samples.")
        else:
            sd.play(full_audio, samplerate=KOKORO_SAMPLERATE)
            sd.wait()
    except Exception as e:
        print(f"kokoro TTS Error: {e}")


def speech_to_text_with_intent(context_prompt: str = "") -> Optional[Dict[str, Any]]:
    samplerate = 16000
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=samplerate) as source:
        print("Listening...")
        try:
            audio_data = r.listen(source, phrase_time_limit=5)
        except sr.WaitTimeoutError:
            print("No speech detected within timeout.")
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
        print("Noise reduction applied.")
        audio_for_stt = reduced_noise
    except Exception as e:
        print(f"Error applying noise reduction: {e}")
        print("Continuing without noise reduction.")
        audio_for_stt = audio_float32

    print("Using OpenAI Whisper for Speech Recognition...")
    input_values = whisper_processor(
        audio_for_stt, sampling_rate=samplerate, return_tensors="pt").input_values
    logits = whisper_model(input_values).logits
    predicted_ids = np.argmax(logits.detach().cpu().numpy(), axis=-1)
    transcription = whisper_processor.batch_decode(
        predicted_ids, skip_special_tokens=True)[0]
    query = transcription
    print(f"Whisper Transcription: {query}")

    if not query.strip():
        print("Empty transcription (silence detected).")
        return {"intent": "silence"}

    user_text = query.lower()
    chat_history.append({"role": "user", "content": query})

    intent_prompt_with_history = f"""
    You are an assistant for a Grab driver. You are friendly, concise, and helpful.
    Your primary goal is to determine the user's intent related to ride bookings and general actions, and respond in JSON format.
    Do not use markdown in your response.

    **Chat History:**
    {[message for message in chat_history]}

    **Context Prompt:**
    {context_prompt}

    **User Text:** "{user_text}"

    Analyze the user text within the context of the chat history and the provided context prompt.
    Determine the user's intent related to ride bookings and general actions.
    Return a JSON object with 'intent' and optionally 'parameters'.
    If you cannot determine the intent, set intent to 'unknown'.

    **Key Intents:**

    1.  **Show Bookings Intent:** Use this intent when the user wants to see their ride bookings OR select a specific ride from a list. Do not shorten place names.
        - **Intent Name:** `"show_bookings"`
        - **Sub-intents and Parameters:**
            - **To show all bookings:** User phrases like "show me my bookings", "my bookings", "list my rides", "what are my bookings".
            - **To select a specific ride:** User phrases like "choose number one", "number 2 please", "I want ride 3", or destination names.
            - **Parameters:**  `"ride_index"` (integer, 1-indexed) OR `"destination"` (string, ride destination)

    2.  **Confirmation Intent:** User confirms an action (e.g., confirming a ride).
        - **Intent Name:** `"confirm"`

    3.  **Decline Intent:** User declines an action (e.g., declining a ride).
        - **Intent Name:** `"decline"`

    4.  **Directions Intent (General):** User requests general directions.
        - **Intent Name:** `"directions"`

    5.  **Show Directions Intent (Ride-Related):** User wants to see directions for a specific booked ride.
        - **Intent Name:** `"show_directions"`

    6.  **Traffic Info Intent:** User requests traffic information.
        - **Intent Name:** `"traffic_info"`

    7.  **Weather Info Intent:** User requests weather information.
        - **Intent Name:** `"weather_info"`

    **Combined Intent Example:** User asks for both weather and traffic.
    If the user says "weather and traffic", return:
    ```json
    {{
      "intent": "combined_info",
      "parameters": {{
        "intents": ["weather_info", "traffic_info"]
      }}
    }}
    ```
    If the user's text matches multiple intents, and it's not a clear combined info request, prioritize the most likely intent or return `"intent": "unknown"`.
    If the user's text does not match any of these intents, and is not a confirmation or decline in the current context, return `"intent": "unknown"`.

    JSON Response:
    """
    try:
        response = model.generate_content(intent_prompt_with_history)
        json_string_raw = response.text
        # Robustly strip markdown and whitespace
        json_string = json_string_raw.replace(
            "```json", "").replace("```", "").strip()
        print(f"Stripped JSON String: {json_string}")  # Debugging
        # Debugging Raw Response
        print(f"LLM Intent JSON (Raw): {json_string_raw}")
        intent_json = json.loads(json_string)
        return intent_json
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}, Raw response: {response.text}")
        return {"intent": "unknown"}
    except Exception as e:
        print(f"LLM Intent Error: {e}")
        return {"intent": "unknown"}


def generate_tts_with_llm(text_prompt: str) -> str:
    try:
        prompt_with_history = f"""
        You are a helpful and concise assistant for a Grab driver.
        Generate a short, natural-sounding spoken response for the driver based on the following:

        **Chat History:**
        {[message for message in chat_history]}

        **Driver Request:**
        {text_prompt}

        Do not use markdown in your response.
        Just provide the plain text response designed to be spoken aloud.
        """
        response = model.generate_content(prompt_with_history)
        llm_generated_text = response.text.strip()
        if llm_generated_text:
            print(f"LLM generated TTS prompt: {llm_generated_text}")
            # Store LLM response in history
            chat_history.append(
                {"role": "assistant", "content": llm_generated_text})
            return llm_generated_text
        else:
            print("LLM did not generate text for TTS.")
            return "Sorry, I encountered an issue. Please try again."  # Fallback TTS message
    except Exception as e:
        print(f"Error generating TTS prompt with LLM: {e}")
        return "Sorry, I encountered an issue. Please try again."  # Fallback TTS message


def show_directions(destination="Airport"):
    tts_prompt_text = f"Showing directions to {destination}."
    tts_prompt = generate_tts_with_llm(tts_prompt_text)
    text_to_speech(tts_prompt)
    print(
        f"Showing directions to {destination} (Placeholder). In a real app, map directions would be displayed.")


def ask_user_to_show_bookings(ride_list: List[str]) -> Optional[str or Dict[str, Any]]:
    if not ride_list:
        tts_prompt = generate_tts_with_llm("No rides available.")
        text_to_speech(tts_prompt)
        return None

    ride_options_text = "\n".join(
        [f"{i+1}. {ride}" for i, ride in enumerate(ride_list)])
    llm_prompt_text = f"Ask the driver to choose a ride from the following options: {ride_options_text}."
    tts_prompt = generate_tts_with_llm(llm_prompt_text)

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
            print(f"Error parsing ride string: {ride}, Error: {e}")
            continue

    while retries >= 0 and chosen_ride is None:
        text_to_speech(tts_prompt)
        context_for_intent = f"The user is choosing a ride from the following options:\n {ride_options_text}. User can say 'number one', 'number two', 'number three' or 'one', 'two', 'three' to choose the ride, or 'show bookings' to see the list again. Or user can say destination name to choose the ride by destination."
        intent_result = speech_to_text_with_intent(
            context_prompt=context_for_intent)
        # Logging Intent Result
        logging.info(f"Intent Result: {intent_result}")

        if intent_result and intent_result["intent"] == "show_bookings":
            if "parameters" in intent_result and "ride_index" in intent_result["parameters"]:
                try:
                    ride_index = intent_result["parameters"].get("ride_index")
                    if ride_index is not None and 1 <= ride_index <= len(ride_list):
                        chosen_ride = ride_list[ride_index - 1]
                        print(f"User chose ride number: {chosen_ride}")
                        return chosen_ride
                    else:
                        print("Invalid ride index chosen.")
                        tts_prompt_invalid_ride = generate_tts_with_llm(
                            "Invalid ride number. Please choose from the list.")
                        text_to_speech(tts_prompt_invalid_ride)
                except (KeyError, TypeError):
                    print("Error parsing ride index from intent.")
                    tts_prompt_choose_again = generate_tts_with_llm(
                        "Please choose the ride number again.")
                    text_to_speech(tts_prompt_choose_again)
            elif "parameters" in intent_result and "destination" in intent_result["parameters"]:
                chosen_destination = intent_result["parameters"].get(
                    "destination")
                ride_index_by_destination = destination_to_ride_index.get(
                    chosen_destination.lower())
                if ride_index_by_destination:
                    chosen_ride = ride_list[ride_index_by_destination - 1]
                    print(f"User chose ride by destination: {chosen_ride}")
                    return chosen_ride
                else:
                    print(
                        f"No ride found for destination: {chosen_destination}")
                    tts_prompt_no_ride_dest = generate_tts_with_llm(
                        "Sorry, I didn't find a ride for that destination. Please choose by number.")
                    text_to_speech(tts_prompt_no_ride_dest)
            else:
                tts_prompt_choose_list = generate_tts_with_llm(
                    "Please choose a ride number or destination from the list.")
                text_to_speech(tts_prompt_choose_list)
                llm_prompt_text_reprompt = "Please choose a ride number or destination from the list."
                tts_prompt = generate_tts_with_llm(llm_prompt_text_reprompt)

        elif intent_result and intent_result["intent"] == "unknown":
            if "raw_text" in intent_result:
                raw_text = intent_result["raw_text"].lower()
                print(f"Unknown intent, raw user text: {raw_text}")
                for dest_lower, ride_index in destination_to_ride_index.items():
                    if dest_lower == raw_text.strip():
                        chosen_ride = ride_list[ride_index - 1]
                        print(
                            f"Interpreted unknown intent as exact destination selection: {chosen_ride}")
                        return chosen_ride
            tts_prompt_unknown = generate_tts_with_llm(
                "Please specify the ride number or destination you're asking about.")
            text_to_speech(tts_prompt_unknown)

        elif intent_result and intent_result["intent"] == "error":
            pass
        elif intent_result and intent_result["intent"] == "silence":
            return None
        elif intent_result and intent_result["intent"] in ["confirm", "decline"]:
            logging.warning(
                f"Ignoring unexpected intent in ride selection: {intent_result['intent']}")
            llm_prompt_text_reprompt_unexpected = "Please choose a ride number or destination from the list."
            tts_prompt = generate_tts_with_llm(
                llm_prompt_text_reprompt_unexpected)

        elif intent_result and intent_result["intent"] == "combined_info":
            if "parameters" in intent_result and "intents" in intent_result["parameters"]:
                intents = intent_result["parameters"]["intents"]
                if "weather_info" in intents:
                    weather_info_text = get_weather_info()
                    weather_info_prompt = generate_tts_with_llm(
                        weather_info_text)
                    text_to_speech(weather_info_prompt)
                if "traffic_info" in intents:
                    traffic_info_text = get_traffic_info()
                    traffic_info_prompt = generate_tts_with_llm(
                        traffic_info_text)
                    text_to_speech(traffic_info_prompt)
        elif intent_result and intent_result["intent"] not in ["show_bookings", "unknown", "error", "silence", "confirm", "decline", "combined_info"]:
            print(
                f"Unexpected intent during ride selection: {intent_result['intent']}")
            return intent_result
        else:
            print(
                f"Unexpected intent result during ride selection: {intent_result}")
            tts_prompt_unexpected_result = generate_tts_with_llm(
                "Please choose a ride number or destination from the list.")
            text_to_speech(tts_prompt_unexpected_result)
            llm_prompt_text_reprompt_result = "Please choose a ride number or destination from the list."
            tts_prompt = generate_tts_with_llm(llm_prompt_text_reprompt_result)

        retries -= 1
        if retries >= 0 and chosen_ride is None:
            if intent_result and intent_result["intent"] != "show_bookings":
                llm_prompt_text_reprompt_list = "Please choose a ride from the list again."
                tts_prompt = generate_tts_with_llm(
                    llm_prompt_text_reprompt_list)

    if chosen_ride:
        return chosen_ride
    else:
        tts_prompt_retry_fail = generate_tts_with_llm(
            "Sorry, please try again.")
        text_to_speech(tts_prompt_retry_fail)
        return None


def read_ride_details_and_confirm(ride_details: str) -> bool:
    details_text = f"Ride details: {ride_details}."
    llm_prompt_text_confirm_details = f"Confirm these ride details: {details_text}. Ask if they want to confirm or have questions (weather/traffic)."
    tts_prompt = generate_tts_with_llm(llm_prompt_text_confirm_details)

    retries = 2
    confirmed = False
    while retries >= 0 and not confirmed:
        text_to_speech(tts_prompt)
        confirmation_intent = speech_to_text_with_intent(
            context_prompt=llm_prompt_text_confirm_details)

        if confirmation_intent and confirmation_intent["intent"] == "confirm":
            confirmed = True
            break
        elif confirmation_intent and confirmation_intent["intent"] == "decline":
            tts_prompt_ride_decline = generate_tts_with_llm("Ride declined.")
            text_to_speech(tts_prompt_ride_decline)
            return False
        elif confirmation_intent and confirmation_intent["intent"] == "silence":
            pass
        elif confirmation_intent and confirmation_intent["intent"] == "unknown":
            tts_prompt_confirm_yes_no = generate_tts_with_llm(
                "Please say yes/no to confirm, or ask weather/traffic.")
            text_to_speech(tts_prompt_confirm_yes_no)
        elif confirmation_intent and confirmation_intent["intent"] == "error":
            pass
        elif confirmation_intent and confirmation_intent["intent"] == "weather_info":
            weather_info_text = get_weather_info()
            weather_info_prompt = generate_tts_with_llm(weather_info_text)
            text_to_speech(weather_info_prompt)
        elif confirmation_intent and confirmation_intent["intent"] == "traffic_info":
            traffic_info_text = get_traffic_info()
            traffic_info_prompt = generate_tts_with_llm(traffic_info_text)
            text_to_speech(traffic_info_prompt)
        elif confirmation_intent and confirmation_intent["intent"] == "combined_info":
            if "parameters" in confirmation_intent and "intents" in confirmation_intent["parameters"]:
                intents = confirmation_intent["parameters"]["intents"]
                if "weather_info" in intents:
                    weather_info_text = get_weather_info()
                    weather_info_prompt = generate_tts_with_llm(
                        weather_info_text)
                    text_to_speech(weather_info_prompt)
                if "traffic_info" in intents:
                    traffic_info_text = get_traffic_info()
                    traffic_info_prompt = generate_tts_with_llm(
                        traffic_info_text)
                    text_to_speech(traffic_info_prompt)
        else:
            tts_prompt_confirm_yes_no_again = generate_tts_with_llm(
                "Please say yes/no to confirm, or ask weather/traffic.")
            text_to_speech(tts_prompt_confirm_yes_no_again)

        retries -= 1
        if retries >= 0 and not confirmed:
            tts_prompt_retry = generate_tts_with_llm(
                "Confirm again: yes/no, or ask weather/traffic.")
            text_to_speech(tts_prompt_retry)
        elif retries < 0 and not confirmed:
            tts_prompt_try_later = generate_tts_with_llm(
                "Sorry, please try again later.")
            text_to_speech(tts_prompt_try_later)
            return False
    return confirmed


def ask_to_show_directions_and_confirm() -> bool:
    llm_prompt_text_directions_yn = "Ask the user if they want to see directions for their ride. Phrase it as a clear yes/no question."
    tts_prompt = generate_tts_with_llm(llm_prompt_text_directions_yn)

    retries = 2
    show_directions_confirmed = False
    while retries >= 0 and not show_directions_confirmed:
        text_to_speech(tts_prompt)
        context_for_directions = "Ask user if they want to show directions (yes/no)."
        directions_intent = speech_to_text_with_intent(
            context_prompt=context_for_directions)

        if directions_intent and directions_intent["intent"] == "show_directions":
            show_directions_confirmed = True
            break
        elif directions_intent and directions_intent["intent"] == "decline":
            tts_prompt_directions_decline = generate_tts_with_llm(
                "Directions declined.")
            text_to_speech(tts_prompt_directions_decline)
            return False
        elif directions_intent and directions_intent["intent"] == "silence":
            pass
        elif directions_intent and directions_intent["intent"] == "unknown":
            tts_prompt_directions_yn_please = generate_tts_with_llm(
                "Do you want to see directions? Please say yes/no.")
            text_to_speech(tts_prompt_directions_yn_please)
        elif directions_intent and directions_intent["intent"] == "error":
            pass
        else:
            tts_prompt_directions_yn_again = generate_tts_with_llm(
                "Do you want to see directions? Please say yes/no.")
            text_to_speech(tts_prompt_directions_yn_again)

        retries -= 1
        if retries >= 0 and not show_directions_confirmed:
            tts_prompt_retry = generate_tts_with_llm(
                "Again: Do you want to see directions?")
            text_to_speech(tts_prompt_retry)
        elif retries < 0 and not show_directions_confirmed:
            tts_prompt_try_later_directions = generate_tts_with_llm(
                "Sorry, please try again later.")
            text_to_speech(tts_prompt_try_later_directions)
            return False
    return show_directions_confirmed


def get_traffic_info():
    return "Traffic is currently light."


def get_weather_info():
    return "The weather is sunny and 25 degrees Celsius."


def get_ride_details_from_text(ride_text: str) -> str:
    parts = ride_text.split(", ")
    if len(parts) >= 2:
        destination = parts[0].replace("Ride to ", "")
        time = parts[1]
        return f"Ride to {destination}, Price: {time}"
    return "Ride details not available."


if __name__ == '__main__':
    try:
        KPipeline(lang_code='a')
        initial_greeting_prompt = generate_tts_with_llm(
            "Hello this is Grab Buddy! How can I help you?")
        text_to_speech(initial_greeting_prompt)
    except RuntimeError:
        print("Kokoro TTS Pipeline likely not initialized in test.")

    print("Grab Driver Buddy Ready! (Ctrl+C to exit)")
    sample_rides = ["Ride to Sunway Universiti from Universiti Tower, RM10",
                    "Ride to KLIA from KLCC, RM60", "Ride to Monash from Sunway Universiti, RM35"]

    last_chosen_destination = None
    ride_confirmed = False

    while True:
        chosen_ride_text = None
        unexpected_intent_result = None
        intent_result_main = speech_to_text_with_intent()
        # Log intent result in main loop
        logging.info(f"Main Loop Intent Result: {intent_result_main}")

        if intent_result_main and intent_result_main["intent"] == "traffic_info":
            traffic_info = get_traffic_info()
            traffic_info_prompt = generate_tts_with_llm(traffic_info)
            text_to_speech(traffic_info_prompt)
            continue

        elif intent_result_main and intent_result_main["intent"] == "weather_info":
            weather_info_text = get_weather_info()
            weather_info_prompt = generate_tts_with_llm(weather_info_text)
            text_to_speech(weather_info_prompt)
            continue
        elif intent_result_main and intent_result_main["intent"] == "combined_info":
            if "parameters" in intent_result_main and "intents" in intent_result_main["parameters"]:
                intents = intent_result_main["parameters"]["intents"]
                if "weather_info" in intents:
                    weather_info_text = get_weather_info()
                    weather_info_prompt = generate_tts_with_llm(
                        weather_info_text)
                    text_to_speech(weather_info_prompt)
                if "traffic_info" in intents:
                    traffic_info_text = get_traffic_info()
                    traffic_info_prompt = generate_tts_with_llm(
                        traffic_info_text)
                    text_to_speech(traffic_info_prompt)
            continue

        elif intent_result_main and intent_result_main["intent"] == "directions":
            destination_for_directions = last_chosen_destination if last_chosen_destination else "Universiti Malaya Faculty of Computer Science"
            show_directions(destination=destination_for_directions)
            continue

        elif intent_result_main and intent_result_main["intent"] == "unknown":
            print("Unknown intent in main loop.")
            # Log when intent is unknown
            logging.info(f"Intent Result (Unknown): {intent_result_main}")
            unknown_prompt = generate_tts_with_llm(
                "Please specify the ride number or destination you're asking about.")
            text_to_speech(unknown_prompt)
            continue

        elif intent_result_main and intent_result_main["intent"] == "error":
            print("Speech recognition error in main loop.")
            continue
        elif intent_result_main and intent_result_main["intent"] == "silence":
            print("Silence detected, listening again...")
            continue
        elif intent_result_main and intent_result_main["intent"] in ["confirm", "decline"]:
            logging.warning(
                f"Ignoring unexpected intent in main loop: {intent_result_main['intent']}")
            continue

        elif not intent_result_main:
            print("No intent result received in main loop.")
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
                print(f"User's ride choice text: {chosen_ride_text}")
                last_chosen_destination = chosen_ride_text.split(" to ")[
                    1].split(",")[0]
                break
            else:
                print("No ride chosen, returning to main loop.")
                continue

        if unexpected_intent_result:
            if unexpected_intent_result["intent"] == "weather_info":
                weather_info_text = get_weather_info()
                weather_info_prompt = generate_tts_with_llm(weather_info_text)
                text_to_speech(weather_info_prompt)
                continue
            elif unexpected_intent_result["intent"] == "traffic_info":
                traffic_info = get_traffic_info()
                traffic_info_prompt = generate_tts_with_llm(traffic_info)
                text_to_speech(traffic_info_prompt)
                continue
            elif unexpected_intent_result["intent"] == "directions":
                destination_for_directions = last_chosen_destination if last_chosen_destination else "Universiti Malaya Faculty of Computer Science"
                show_directions(destination=destination_for_directions)
                continue
            else:
                print(
                    f"Unhandled unexpected intent: {unexpected_intent_result['intent']}")
                unknown_action_prompt = generate_tts_with_llm(
                    "Sorry, I cannot do that right now.")
                text_to_speech(unknown_action_prompt)
                continue

        if not chosen_ride_text:
            continue

        sample_ride_details = get_ride_details_from_text(chosen_ride_text)
        ride_confirmed = read_ride_details_and_confirm(sample_ride_details)
        print(f"Ride confirmed: {ride_confirmed}")

        if not ride_confirmed:
            continue

        show_directions_for_ride = ask_to_show_directions_and_confirm()
        print(f"Show directions (ride related): {show_directions_for_ride}")

        final_confirmation_prompt = generate_tts_with_llm(
            "Ride booked. Thank you!")
        text_to_speech(final_confirmation_prompt)
        print("Ride booking flow completed. Listening for next command.\n")
        chat_history = []  # Clear chat history after completing a flow
