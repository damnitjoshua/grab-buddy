"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { DriverStatus, type RideStatus } from "./types";

interface UseVoiceCommandsProps {
  driverStatus: DriverStatus;
  rideStatus: RideStatus;
  toggleDriverStatus: () => void;
  acceptRide: () => void;
  declineRide: () => void;
  startRide: () => void;
  endRide: () => void;
  toggleQueuedRidesPanel: () => void;
  onVoiceCommand: (transcript: string) => void;
}

declare var SpeechRecognition: any;
declare var webkitSpeechRecognition: any;

export function useVoiceCommands({
  driverStatus,
  rideStatus,
  toggleDriverStatus,
  acceptRide,
  declineRide,
  startRide,
  endRide,
  toggleQueuedRidesPanel,
  onVoiceCommand,
}: UseVoiceCommandsProps) {
  const [isSupported, setIsSupported] = useState(false);
  const [isListeningForWakeWord, setIsListeningForWakeWord] = useState(false);
  const [isVoiceCommandActive, setIsVoiceCommandActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wakeWordRecognitionRef = useRef<SpeechRecognition | null>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);

  // Audio streaming refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const audioBufferRef = useRef<Float32Array[]>([]);
  const accumulatedSamplesRef = useRef<number>(0);
  const sampleRateRef = useRef<number>(16000); // Default, will be updated with actual context
  const chunkDuration = 10; // seconds
  const bufferSize = 4096; // ScriptProcessor buffer size


  useEffect(() => {
    if (
      typeof window !== "undefined" &&
      ("SpeechRecognition" in window || "webkitSpeechRecognition" in window)
    ) {
      setIsSupported(true);
      const SpeechRecognition =
        window.SpeechRecognition || webkitSpeechRecognition;

      wakeWordRecognitionRef.current = new SpeechRecognition();
      wakeWordRecognitionRef.current.continuous = true;
      wakeWordRecognitionRef.current.interimResults = false;

      startWakeWordDetection();
    } else {
      setError("Speech recognition is not supported in this browser");
    }

    return () => {
      stopWakeWordDetection();
      stopAudioStreaming();
      if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
        websocketRef.current.close();
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  const startWakeWordDetection = useCallback(() => {
    if (!wakeWordRecognitionRef.current) return;

    try {
      wakeWordRecognitionRef.current.onresult = (event) => {
        const transcript = event.results[event.results.length - 1][0].transcript
          .toLowerCase()
          .trim();
        console.log("Wake word detection heard:", transcript);

        if (transcript.includes("hey grab")) {
          console.log("Wake word detected!");
          stopWakeWordDetection();
          speakFeedback("Voice commands activated.");
          startAudioStreaming(); // Start audio streaming after wake word
        }
      };

      wakeWordRecognitionRef.current.onend = () => {
        if (isListeningForWakeWord) {
          console.log("Wake word detection ended, restarting...");
          wakeWordRecognitionRef.current?.start();
        }
      };

      wakeWordRecognitionRef.current.onerror = (event) => {
        console.error("Wake word detection error:", event.error);
        if (event.error !== "no-speech") {
          setError(`Wake word detection error: ${event.error}`);
        }
        if (isListeningForWakeWord) {
          setTimeout(() => {
            wakeWordRecognitionRef.current?.start();
          }, 100);
        }
      };

      wakeWordRecognitionRef.current.start();
      setIsListeningForWakeWord(true);
      setError(null);
    } catch (err) {
      console.error("Error starting wake word detection:", err);
      setError("Failed to start wake word detection");
    }
  }, [isListeningForWakeWord]);

  const stopWakeWordDetection = useCallback(() => {
    if (!wakeWordRecognitionRef.current) return;
    try {
      wakeWordRecognitionRef.current.stop();
      setIsListeningForWakeWord(false);
    } catch (err) {
      console.error("Error stopping wake word detection:", err);
    }
  }, []);

  const speakFeedback = useCallback((text: string) => {
    if ("speechSynthesis" in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  }, []);

  const startAudioStreaming = useCallback(async () => {
    try {
      console.log("Starting streaming...");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log("getUserMedia success", stream);
      localStreamRef.current = stream;

      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      console.log("AudioContext sampleRate:", audioContext.sampleRate);
      sampleRateRef.current = audioContext.sampleRate;
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

      const socket = new WebSocket("ws://127.0.0.1:8000/ws/grab_buddy"); // Replace with your server URL
      socket.binaryType = "arraybuffer";

      // Initialize state
      setIsVoiceCommandActive(true);
      audioBufferRef.current = [];
      accumulatedSamplesRef.current = 0;
      websocketRef.current = socket;
      audioContextRef.current = audioContext;
      processorRef.current = processor;
      sourceRef.current = source;


      processor.onaudioprocess = (e) => {
        const inputBuffer = e.inputBuffer.getChannelData(0);
        const buffer44100Hz = new Float32Array(inputBuffer);

        // *** WEB AUDIO API RESAMPLING START ***
        const originalSampleRate = audioContext.sampleRate;
        const targetSampleRate = 16000;
        let buffer16000Hz;

        if (originalSampleRate !== targetSampleRate) {
          console.log(`Resampling from ${originalSampleRate}Hz to ${targetSampleRate}Hz...`);
          const offlineCtx = new OfflineAudioContext(
            1,
            inputBuffer.length * (targetSampleRate / originalSampleRate),
            targetSampleRate
          );
          const bufferSource = offlineCtx.createBufferSource();
          const audioBuffer = offlineCtx.createBuffer(1, inputBuffer.length, originalSampleRate);
          audioBuffer.copyToChannel(inputBuffer, 0);
          bufferSource.buffer = audioBuffer;
          bufferSource.connect(offlineCtx.destination);
          bufferSource.start();

          offlineCtx
            .startRendering()
            .then((renderedBuffer) => {
              buffer16000Hz = renderedBuffer.getChannelData(0);

              // *** MOVED ORIGINAL CHUNKING AND SENDING LOGIC HERE, USING buffer16000Hz ***
              audioBufferRef.current.push(buffer16000Hz);
              accumulatedSamplesRef.current += buffer16000Hz.length;

              // Calculate how many samples we need for chunkDuration seconds
              const samplesNeeded = targetSampleRate * chunkDuration;

              if (accumulatedSamplesRef.current >= samplesNeeded) {
                // Create a single buffer with exactly chunkDuration seconds of audio
                const combinedBuffer = new Float32Array(samplesNeeded);
                let offset = 0;
                let remainingSamples = samplesNeeded;

                // Process buffers until we have exactly chunkDuration seconds
                while (remainingSamples > 0 && audioBufferRef.current.length > 0) {
                  const currentBuffer = audioBufferRef.current[0];
                  const samplesToCopy = Math.min(currentBuffer.length, remainingSamples);

                  combinedBuffer.set(currentBuffer.subarray(0, samplesToCopy), offset);
                  offset += samplesToCopy;
                  remainingSamples -= samplesToCopy;

                  if (samplesToCopy === currentBuffer.length) {
                    audioBufferRef.current.shift();
                  } else {
                    // If we didn't use the entire buffer, keep the remainder
                    audioBufferRef.current[0] = currentBuffer.subarray(samplesToCopy);
                  }
                }

                // Update the accumulated samples count
                accumulatedSamplesRef.current = audioBufferRef.current.reduce((acc, buf) => acc + buf.length, 0);

                // Send the chunk if WebSocket is open
                if (socket.readyState === WebSocket.OPEN) {
                  socket.send(combinedBuffer.buffer);
                  console.log(`Sent audio chunk of ${combinedBuffer.length} samples (${chunkDuration} seconds)`);
                }
              }
              // *** END MOVED CHUNKING AND SENDING LOGIC ***
            })
            .catch((err) => {
              console.error("Rendering failed: " + err);
              buffer16000Hz = buffer44100Hz; // Fallback to original buffer in case of resampling error
              alert("Audio resampling failed! Audio might be slowed down. Please check console for errors.");
            });
          return; // Important: Exit here as processing continues in then()
        } else {
          buffer16000Hz = buffer44100Hz; // No resampling needed - unlikely in browsers

          // *** ORIGINAL CHUNKING AND SENDING LOGIC - NOW USING buffer16000Hz (which is buffer44100Hz if no resampling) ***
          audioBufferRef.current.push(buffer16000Hz);
          accumulatedSamplesRef.current += buffer16000Hz.length;

          // Calculate how many samples we need for chunkDuration seconds
          const samplesNeeded = sampleRateRef.current * chunkDuration;

          if (accumulatedSamplesRef.current >= samplesNeeded) {
            // Create a single buffer with exactly chunkDuration seconds of audio
            const combinedBuffer = new Float32Array(samplesNeeded);
            let offset = 0;
            let remainingSamples = samplesNeeded;

            // Process buffers until we have exactly chunkDuration seconds
            while (remainingSamples > 0 && audioBufferRef.current.length > 0) {
              const currentBuffer = audioBufferRef.current[0];
              const samplesToCopy = Math.min(currentBuffer.length, remainingSamples);

              combinedBuffer.set(currentBuffer.subarray(0, samplesToCopy), offset);
              offset += samplesToCopy;
              remainingSamples -= samplesToCopy;

              if (samplesToCopy === currentBuffer.length) {
                audioBufferRef.current.shift();
              } else {
                // If we didn't use the entire buffer, keep the remainder
                audioBufferRef.current[0] = currentBuffer.subarray(samplesToCopy);
              }
            }

            // Update the accumulated samples count
            accumulatedSamplesRef.current = audioBufferRef.current.reduce((acc, buf) => acc + buf.length, 0);

            // Send the chunk if WebSocket is open
            if (socket.readyState === WebSocket.OPEN) {
              socket.send(combinedBuffer.buffer);
              console.log(`Sent audio chunk of ${combinedBuffer.length} samples (${chunkDuration} seconds)`);
            }
          }
          // *** END ORIGINAL CHUNKING AND SENDING LOGIC  ***
        }
        // *** WEB AUDIO API RESAMPLING END ***
      };

      socket.onopen = () => {
        console.log("WebSocket connection opened for audio streaming");
        source.connect(processor);
        processor.connect(audioContext.destination);
      };

      socket.onerror = (error) => {
        console.error("WebSocket error:", error);
        setError("WebSocket error during audio streaming");
        setIsVoiceCommandActive(false);
      };

      socket.onclose = () => {
        console.log("WebSocket connection closed for audio streaming");
        setIsVoiceCommandActive(false);
      };


    } catch (err) {
      console.error("Error starting audio streaming:", err);
      setError("Failed to start audio streaming");
      setIsVoiceCommandActive(false);
    }
  }, []);

  const stopAudioStreaming = useCallback(() => {
    console.log("Stopping streaming...");
    setIsVoiceCommandActive(false);

    if (processorRef.current) {
      processorRef.current.onaudioprocess = null;
      processorRef.current.disconnect();
    }
    if (sourceRef.current) {
      sourceRef.current.disconnect();
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (websocketRef.current) {
      websocketRef.current.close();
      websocketRef.current = null;
    }

    audioBufferRef.current = [];
    accumulatedSamplesRef.current = 0;
    console.log("Streaming stopped");
  }, []);


  const toggleVoiceActivation = useCallback(() => {
    if (isVoiceCommandActive) {
      speakFeedback("Voice commands deactivated.");
      stopAudioStreaming();
      startWakeWordDetection();
    } else {
      stopWakeWordDetection();
      startAudioStreaming();
      speakFeedback("Voice commands activated.");
    }
  }, [isVoiceCommandActive, startWakeWordDetection, speakFeedback, stopAudioStreaming, startAudioStreaming]);

  return {
    isSupported,
    isListeningForWakeWord,
    isVoiceCommandActive,
    error,
    toggleVoiceActivation,
  };
}