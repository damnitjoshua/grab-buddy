import asyncio
import json
import logging
import os
import uuid
import wave
import numpy as np

import sounddevice as sd
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay, MediaPlayer, MediaRecorder
import websockets
from websockets.server import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pcs = set()
audio_track = None
audio_recorder = None

# 🔧 [ADDED] Function to handle raw binary audio messages
async def process_raw_audio_bytes(data_bytes):
    try:
        # Ensure the buffer size is divisible by the element size (2 bytes for int16)
        num_samples = len(data_bytes) // 2  # Dividing by 2 since int16 is 2 bytes per sample
        if len(data_bytes) % 2 != 0:
            logger.warning(f"Audio buffer size is not a multiple of element size. Trimming excess data.")
            data_bytes = data_bytes[:num_samples * 2]  # Trim excess data

        # Convert byte data to numpy array of int16 type (16-bit audio samples)
        audio_np = np.frombuffer(data_bytes, dtype=np.int16)

        # Sample rate should match what your frontend sends (e.g., 16000 Hz)
        sample_rate = 16000  # Adjust to match frontend configuration
        await process_audio_chunk_for_ai(audio_np, sample_rate)
    except Exception as e:
        logger.error(f"Error processing raw audio bytes: {e}")
        
async def process_audio_chunk_for_ai(audio_chunk_np, sample_rate):
    """
    Placeholder function to send audio chunk to AI backend for processing.
    In a real application, this would be replaced with actual integration logic,
    e.g., sending data over a queue, shared memory, or network socket to the AI backend.
    """
    print(f"Received audio chunk for AI processing - shape: {audio_chunk_np.shape}, rate: {sample_rate}")
    # Integration with your AI backend logic goes here


async def run_signaling_server():
    global audio_track, audio_recorder

    async def handle_offer(offer_sdp, ws):
        pc = RTCPeerConnection()
        pc_id = "PeerConnection-%s" % uuid.uuid4()
        pcs.add(pc)

        def log_info(msg, *args):
            logger.info(pc_id + " " + msg, *args)

        log_info("Created for %s", ws.remote_address)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log_info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                pcs.discard(pc)

        @pc.on("track")
        def on_track(track):
            log_info("Track %s received", track.kind)
            if track.kind == "audio":
                audio_track = track

                async def relay_audio():
                    nonlocal audio_track
                    while True:
                        try:
                            audio_chunk = await audio_track.recv()
                            if audio_chunk:
                                audio_np = audio_chunk.to_ndarray()
                                sample_rate = audio_track.sample_rate
                                asyncio.create_task(process_audio_chunk_for_ai(audio_np, sample_rate))
                        except Exception as e:
                            logger.error(f"Error receiving or processing audio chunk: {e}")
                            break

                log_info("Starting audio stream processing for AI...")
                asyncio.ensure_future(relay_audio())

            @track.on("ended")
            async def on_ended():
                log_info("Track %s ended", track.kind)

        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
        await pc.setRemoteDescription(offer)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        payload = {"type": "answer", "sdp": pc.localDescription.sdp}
        await ws.send(json.dumps(payload))

    async def handle_candidate(candidate_data, pc):
        logger.info(f"Received candidate data: {candidate_data}")
        try:
            candidate_str = candidate_data["candidate"]["candidate"]
            candidate = RTCIceCandidate(
                candidate_str,
                foundation=candidate_data["candidate"]["foundation"],
                ip=candidate_data["candidate"]["ip"],
                port=candidate_data["candidate"]["port"],
                priority=candidate_data["candidate"]["priority"],
                protocol=candidate_data["candidate"]["protocol"],
                type=candidate_data["candidate"]["type"],
                sdpMid=candidate_data["candidate"]["sdpMid"],
                sdpMLineIndex=candidate_data["candidate"]["sdpMLineIndex"],
            )
            await pc.addIceCandidate(candidate)
        except Exception as e:
            logger.error(f"Error creating RTCIceCandidate: {e}")

    async def ws_handler(websocket):
        pc = None

        try:
            async for message in websocket:
                try:
                    # 🔧 [CHANGED] Handle binary (audio) messages
                    if isinstance(message, bytes):
                        logger.info(f"Received binary audio message of length {len(message)}")
                        await process_raw_audio_bytes(message)
                        continue

                    # Handle JSON signaling messages
                    data = json.loads(message)
                    message_type = data["type"]

                    if message_type == "offer":
                        await handle_offer(data["sdp"], websocket)
                    elif message_type == "candidate":
                        if pc is None:
                            pcs_list = list(pcs)
                            if pcs_list:
                                pc = pcs_list[-1]
                            else:
                                logger.warning("Received candidate before offer, ignoring.")
                                continue
                        await handle_candidate(data["candidate"], pc)

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    break

        finally:
            if pc and pc in pcs:
                await pc.close()
                pcs.discard(pc)

    async with serve(ws_handler, "0.0.0.0", 8765):
        print("WebSocket signaling server started at ws://0.0.0.0:8765")
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(run_signaling_server())
    except KeyboardInterrupt:
        pass
    finally:
        print("\nExiting")
