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
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null); // (not used anymore)
  const localStreamRef = useRef<MediaStream | null>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null); // 🟩 NEW

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
          startAudioStreaming(); // 🟩 Stream audio after wake word
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

  // 🟩 REPLACED startCommandRecognition with audio streaming
  const startAudioStreaming = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      localStreamRef.current = stream;

      // 🟩 Setup WebSocket
      websocketRef.current = new WebSocket("ws://localhost:8765");

      websocketRef.current.onopen = () => {
        console.log("WebSocket connection opened");
        // 🟩 Start sending audio chunks
        const mediaRecorder = new MediaRecorder(stream, {
          mimeType: "audio/webm;codecs=opus",
        });
        mediaRecorderRef.current = mediaRecorder;

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0 && websocketRef.current?.readyState === WebSocket.OPEN) {
            websocketRef.current.send(event.data); // 🟩 Send binary audio
          }
        };

        mediaRecorder.start(250); // send chunks every 250ms
        setIsVoiceCommandActive(true);
      };

      websocketRef.current.onerror = (error) => {
        console.error("WebSocket error:", error);
        setError("WebSocket error");
      };

      websocketRef.current.onclose = () => {
        console.log("WebSocket connection closed");
      };
    } catch (err) {
      console.error("Error starting audio streaming:", err);
      setError("Failed to start audio streaming");
    }
  }, []);

  const stopAudioStreaming = useCallback(() => {
    if (mediaRecorderRef.current) {
      console.log("stopAudioStreaming: mediaRecorderRef.current is NOT null"); // 🟢 DEBUG - Check if this line is printed
      mediaRecorderRef.current.stop(); // 🟩 Stop media recorder
      mediaRecorderRef.current.ondataavailable = async (event) => { // Capture the last data
        console.log("ondataavailable event triggered"); // 🟢 DEBUG - Check if this line is printed
        if (event.data && event.data.size > 0) {
          console.log("event.data.size > 0, downloading Blob"); // 🟢 DEBUG - Check if this line is printed
          downloadBlob(event.data, "recorded-audio.webm"); // Download the Blob
        } else {
          console.log("event.data is empty or size is 0"); // 🟢 DEBUG - Check if this line is printed
        }
  
        localStreamRef.current?.getTracks().forEach((track) => track.stop());
  
        if (websocketRef.current?.readyState === WebSocket.OPEN) {
          websocketRef.current.close();
        }
  
        mediaRecorderRef.current = null;
        localStreamRef.current = null;
        websocketRef.current = null;
        setIsVoiceCommandActive(false);
      };
    } else {
      console.log("stopAudioStreaming: mediaRecorderRef.current is NULL"); // 🟢 DEBUG - Check if this line is printed
      localStreamRef.current?.getTracks().forEach((track) => track.stop());
      if (websocketRef.current?.readyState === WebSocket.OPEN) {
        websocketRef.current.close();
      }
      mediaRecorderRef.current = null;
      localStreamRef.current = null;
      websocketRef.current = null;
      setIsVoiceCommandActive(false);
    }
  }, []);
  
  // Helper function to download Blob
  const downloadBlob = (blob: Blob, filename: string) => {
    console.log("downloadBlob function called", { blob, filename }); // 🟢 DEBUG - Check if this line is printed
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    console.log("downloadBlob function finished"); // 🟢 DEBUG - Check if this line is printed
  };

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
  }, [isVoiceCommandActive, startWakeWordDetection, speakFeedback]);

  return {
    isSupported,
    isListeningForWakeWord,
    isVoiceCommandActive,
    error,
    toggleVoiceActivation,
  };
}
