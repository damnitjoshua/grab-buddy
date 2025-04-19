"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { DriverStatus, RideStatus } from "./types";

declare var SpeechRecognition: any;
declare var webkitSpeechRecognition: any;

interface UseVoiceCommandsProps {
	driverStatus: DriverStatus;
	rideStatus: RideStatus;
	setDriverStatus: (status: DriverStatus) => void;
	setRideStatus: (status: RideStatus) => void;
	toggleDriverStatus: () => void;
	acceptRide: () => void;
	declineRide: () => void;
	startRide: () => void;
	endRide: () => void;
	toggleQueuedRidesPanel: () => void;
}

export function useVoiceCommands({
	driverStatus,
	rideStatus,
	toggleDriverStatus,
	acceptRide,
	declineRide,
	startRide,
	endRide,
	toggleQueuedRidesPanel,
}: UseVoiceCommandsProps) {
	const [isSupported, setIsSupported] = useState(false);
	const [isListeningForWakeWord, setIsListeningForWakeWord] = useState(false);
	const [isVoiceCommandActive, setIsVoiceCommandActive] = useState(false);
	const [lastCommand, setLastCommand] = useState("");
	const [error, setError] = useState<string | null>(null);

	const wakeWordRecognitionRef = useRef<SpeechRecognition | null>(null);
	const commandRecognitionRef = useRef<SpeechRecognition | null>(null);
	const isCommandRecognitionRunningRef = useRef<boolean>(false);
	const isStartingCommandRecognitionRef = useRef<boolean>(false);

	const socketRef = useRef<WebSocket | null>(null);
	const audioContextRef = useRef<AudioContext | null>(null);
	const processorRef = useRef<ScriptProcessorNode | null>(null);
	const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
	const audioBufferRef = useRef<Float32Array[]>([]);
	const accumulatedSamplesRef = useRef<number>(0);
	const sampleRateRef = useRef<number>(16000);
	const playbackCountRef = useRef(0);
	const isPlayingRef = useRef(false);
	const isTTSSpeakingRef = useRef(false);
	const chunkDuration = 6;
	const bufferSize = 4096;

	useEffect(() => {
		if (typeof window !== "undefined" && ("SpeechRecognition" in window || "webkitSpeechRecognition" in window)) {
			setIsSupported(true);
			const SpeechRecognition = window.SpeechRecognition || webkitSpeechRecognition;
			wakeWordRecognitionRef.current = new SpeechRecognition();
			wakeWordRecognitionRef.current.continuous = true;
			wakeWordRecognitionRef.current.interimResults = false;

			commandRecognitionRef.current = new SpeechRecognition();
			commandRecognitionRef.current.continuous = false;
			commandRecognitionRef.current.interimResults = false;

			startWakeWordDetection();
		} else {
			setError("Speech recognition is not supported in this browser");
		}

		return () => {
			stopWakeWordDetection();
			stopCommandRecognition();
			stopStreaming();
		};
	}, []);

	useEffect(() => {
		if (isVoiceCommandActive) {
			startStreaming();
		} else {
			stopStreaming();
		}
	}, [isVoiceCommandActive]);

	const startWakeWordDetection = useCallback(() => {
		if (!wakeWordRecognitionRef.current) {
			const SpeechRecognition = window.SpeechRecognition || webkitSpeechRecognition;
			wakeWordRecognitionRef.current = new SpeechRecognition();
			wakeWordRecognitionRef.current.continuous = true;
			wakeWordRecognitionRef.current.interimResults = false;
		}

		wakeWordRecognitionRef.current.onresult = (event) => {
			const transcript = event.results[event.results.length - 1][0].transcript.toLowerCase().trim();
			if (transcript.includes("hey grab")) {
				stopWakeWordDetection();
				startCommandRecognition();
			}
		};

		wakeWordRecognitionRef.current.onend = () => {
			if (isListeningForWakeWord) {
				wakeWordRecognitionRef.current?.start();
			}
		};

		wakeWordRecognitionRef.current.onerror = (event) => {
			if (event.error !== "no-speech") {
				setError(`Wake word error: ${event.error}`);
			}
			if (isListeningForWakeWord) {
				setTimeout(() => wakeWordRecognitionRef.current?.start(), 100);
			}
		};

		wakeWordRecognitionRef.current.start();
		setIsListeningForWakeWord(true);
	}, [isListeningForWakeWord]);

	const stopWakeWordDetection = useCallback(() => {
		if (wakeWordRecognitionRef.current) {
			wakeWordRecognitionRef.current.stop();
			wakeWordRecognitionRef.current = null;
		}
		setIsListeningForWakeWord(false);
	}, []);

	const startCommandRecognition = useCallback(() => {
		if (!commandRecognitionRef.current) return;
		if (isCommandRecognitionRunningRef.current || isStartingCommandRecognitionRef.current) return;

		isStartingCommandRecognitionRef.current = true;

		commandRecognitionRef.current.onstart = () => {
			isCommandRecognitionRunningRef.current = true;
			isStartingCommandRecognitionRef.current = false;
		};

		commandRecognitionRef.current.onresult = (event) => {
			const transcript = event.results[0][0].transcript.toLowerCase().trim();
			setLastCommand(transcript);
			// No local command processing — backend handles it
		};

		commandRecognitionRef.current.onend = () => {
			isCommandRecognitionRunningRef.current = false;
			isStartingCommandRecognitionRef.current = false;
			if (isVoiceCommandActive) {
				setTimeout(() => {
					if (!isCommandRecognitionRunningRef.current && !isStartingCommandRecognitionRef.current) {
						commandRecognitionRef.current?.start();
					}
				}, 200);
			}
		};

		commandRecognitionRef.current.onerror = (event) => {
			isCommandRecognitionRunningRef.current = false;
			isStartingCommandRecognitionRef.current = false;
			if (event.error !== "no-speech") setError(`Command error: ${event.error}`);
			if (isVoiceCommandActive) {
				setTimeout(() => {
					if (!isCommandRecognitionRunningRef.current && !isStartingCommandRecognitionRef.current) {
						commandRecognitionRef.current?.start();
					}
				}, 100000000000);
			}
		};

		commandRecognitionRef.current.start();
		setIsVoiceCommandActive(true);
	}, [isVoiceCommandActive]);

	const stopCommandRecognition = useCallback(() => {
		if (!commandRecognitionRef.current) return;
		commandRecognitionRef.current.stop();
		setIsVoiceCommandActive(false);
		isCommandRecognitionRunningRef.current = false;
		isStartingCommandRecognitionRef.current = false;
	}, []);

	const startStreaming = async () => {
		try {
			const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
			const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
			sampleRateRef.current = audioContext.sampleRate;
			const source = audioContext.createMediaStreamSource(stream);
			const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);
			const socket = new WebSocket("ws://127.0.0.1:8000/ws/grab_buddy");
			socket.binaryType = "arraybuffer";

			processor.onaudioprocess = (e) => {
				if (isTTSSpeakingRef.current) return;
				const inputBuffer = e.inputBuffer.getChannelData(0);
				const buffer44100Hz = new Float32Array(inputBuffer);
				const originalSampleRate = audioContext.sampleRate;
				const targetSampleRate = 16000;

				if (originalSampleRate !== targetSampleRate) {
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

					offlineCtx.startRendering().then((renderedBuffer) => {
						const buffer16000Hz = renderedBuffer.getChannelData(0);
						handleChunkingAndSend(buffer16000Hz, socket, targetSampleRate);
					});
				} else {
					handleChunkingAndSend(buffer44100Hz, socket, sampleRateRef.current);
				}
			};

			socket.onopen = () => {
				source.connect(processor);
				processor.connect(audioContext.destination);
				socketRef.current = socket;
				audioContextRef.current = audioContext;
				processorRef.current = processor;
				sourceRef.current = source;
			};

			socket.onmessage = (event) => {
				console.log("WebSocket message received:", event.data);
				if (typeof event.data === "string") {
					try {
						const message = JSON.parse(event.data);
						if (message.type === "state_update") {
							const { driverStatus: newDriverStatus, rideStatus: newRideStatus } = message.payload;

							console.log(newDriverStatus);

							// Handle driver status change
							if (newDriverStatus) {
								if (newDriverStatus === "ONLINE") {
									toggleDriverStatus();
								} else if (newDriverStatus === "OFFLINE") {
toggleDriverStatus();
								}
							}

							// Handle ride status change
							if (newRideStatus) {
								switch (newRideStatus) {
									case "ACCEPTED":
										acceptRide();
										break;
									case "DECLINED":
										declineRide();
										break;
									case "STARTED":
										startRide();
										break;
									case "ENDED":
										endRide();
										break;
									default:
										break;
								}
							}
						}
					} catch (e) {
						console.warn("Non-JSON message received:", event.data);
					}
					return;
				}

				// Handle audio playback
				const arrayBuffer = event.data;
				if (!audioContextRef.current) return;
				audioContextRef.current.decodeAudioData(arrayBuffer.slice(0), (decodedData) => {
					const source = audioContextRef.current!.createBufferSource();
					isPlayingRef.current = true;
					isTTSSpeakingRef.current = true;
					source.buffer = decodedData;
					source.connect(audioContextRef.current.destination);
					source.start();
					source.onended = () => {
						isPlayingRef.current = false;
						isTTSSpeakingRef.current = false;
						playbackCountRef.current++;
						if (playbackCountRef.current >= 2) stopStreaming();
					};
				});
			};

			socket.onclose = () => {
				isPlayingRef.current = false;
				isTTSSpeakingRef.current = false;
				stopCommandRecognition();
				startWakeWordDetection();
				stopStreaming();
			};

			socket.onerror = () => {
				isPlayingRef.current = false;
				isTTSSpeakingRef.current = false;
				stopCommandRecognition();
				startWakeWordDetection();
				stopStreaming();
			};
		} catch (err) {
			console.error("Streaming error:", err);
		}
	};

	const handleChunkingAndSend = (buffer: Float32Array, socket: WebSocket, sampleRate: number) => {
		audioBufferRef.current.push(buffer);
		accumulatedSamplesRef.current += buffer.length;
		const samplesNeeded = sampleRate * chunkDuration;

		if (accumulatedSamplesRef.current >= samplesNeeded) {
			const combinedBuffer = new Float32Array(samplesNeeded);
			let offset = 0;
			let remaining = samplesNeeded;

			while (remaining > 0 && audioBufferRef.current.length > 0) {
				const current = audioBufferRef.current[0];
				const toCopy = Math.min(current.length, remaining);
				combinedBuffer.set(current.subarray(0, toCopy), offset);
				offset += toCopy;
				remaining -= toCopy;

				if (toCopy === current.length) {
					audioBufferRef.current.shift();
				} else {
					audioBufferRef.current[0] = current.subarray(toCopy);
				}
			}

			accumulatedSamplesRef.current = audioBufferRef.current.reduce((acc, buf) => acc + buf.length, 0);
			if (socket.readyState === WebSocket.OPEN) {
				socket.send(combinedBuffer.buffer);
			}
		}
	};

	const stopStreaming = () => {
		if (isPlayingRef.current || isTTSSpeakingRef.current) return;
		processorRef.current?.disconnect();
		sourceRef.current?.disconnect();
		if (audioContextRef.current && audioContextRef.current.state !== "closed") {
			audioContextRef.current.close();
		}
		audioContextRef.current = null;
		socketRef.current?.close();
		socketRef.current = null;
		audioBufferRef.current = [];
		accumulatedSamplesRef.current = 0;
		playbackCountRef.current = 0;
	};

	return {
		isSupported,
		isListeningForWakeWord,
		isVoiceCommandActive,
		lastCommand,
		error,
	};
}
