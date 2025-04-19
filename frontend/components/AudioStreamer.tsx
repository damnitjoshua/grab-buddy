"use client";
import { useRef, useState, useEffect } from "react";

const AudioStreamer: React.FC = () => {
	const socketRef = useRef<WebSocket | null>(null);
	const audioContextRef = useRef<AudioContext | null>(null);
	const processorRef = useRef<ScriptProcessorNode | null>(null);
	const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
	const [isStreaming, setIsStreaming] = useState(false);
	const chunkDuration = 10; // seconds
	const bufferSize = 4096; // ScriptProcessor buffer size
	const audioBufferRef = useRef<Float32Array[]>([]);
	const accumulatedSamplesRef = useRef<number>(0);
	const sampleRateRef = useRef<number>(16000); // Default, will be updated with actual context

	useEffect(() => {
		return () => {
			if (socketRef.current) {
				socketRef.current.close();
			}
			if (audioContextRef.current) {
				audioContextRef.current.close();
			}
		};
	}, []);

	const startStreaming = async () => {
		try {
			console.log("Starting streaming...");
			const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
			console.log("getUserMedia success", stream);

			const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
			console.log("AudioContext sampleRate:", audioContext.sampleRate);
			sampleRateRef.current = audioContext.sampleRate;
			const source = audioContext.createMediaStreamSource(stream);
			const processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

			const socket = new WebSocket("ws://127.0.0.1:8000/ws/grab_buddy");
			socket.binaryType = "arraybuffer";

			// Initialize state
			setIsStreaming(true);
			audioBufferRef.current = [];
			accumulatedSamplesRef.current = 0;

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

							// Calculate how many samples we need for 5 seconds
							const samplesNeeded = targetSampleRate * chunkDuration;

							if (accumulatedSamplesRef.current >= samplesNeeded) {
								// Create a single buffer with exactly 5 seconds of audio
								const combinedBuffer = new Float32Array(samplesNeeded);
								let offset = 0;
								let remainingSamples = samplesNeeded;

								// Process buffers until we have exactly 5 seconds
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
							// You might want to handle fallback chunk processing here if needed
							alert("Audio resampling failed! Audio might be slowed down. Please check console for errors.");
						});
					return; // Important: Exit here as processing continues in then()
				} else {
					buffer16000Hz = buffer44100Hz; // No resampling needed - unlikely in browsers

					// *** ORIGINAL CHUNKING AND SENDING LOGIC - NOW USING buffer16000Hz (which is buffer44100Hz if no resampling) ***
					audioBufferRef.current.push(buffer16000Hz);
					accumulatedSamplesRef.current += buffer16000Hz.length;

					// Calculate how many samples we need for 5 seconds
					const samplesNeeded = sampleRateRef.current * chunkDuration;

					if (accumulatedSamplesRef.current >= samplesNeeded) {
						// Create a single buffer with exactly 5 seconds of audio
						const combinedBuffer = new Float32Array(samplesNeeded);
						let offset = 0;
						let remainingSamples = samplesNeeded;

						// Process buffers until we have exactly 5 seconds
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
				console.log("WebSocket connection opened");
				source.connect(processor);
				processor.connect(audioContext.destination);

				socketRef.current = socket;
				audioContextRef.current = audioContext;
				processorRef.current = processor;
				sourceRef.current = source;
			};

			socket.onmessage = (event) => {
				console.log("Received message from server");
				const arrayBuffer = event.data;
				console.log("Message data:", arrayBuffer);

				audioContext.decodeAudioData(
					arrayBuffer.slice(0) as ArrayBuffer,
					(decodedData) => {
						console.log("Audio data decoded successfully");
						const source = audioContext.createBufferSource();
						source.buffer = decodedData;
						source.connect(audioContext.destination);
						source.start();
						console.log("Playback started");
					},
					(e) => {
						console.error("Error decoding audio data", e);
					}
				);
			};

			socket.onclose = () => {
				console.log("WebSocket connection closed");
				setIsStreaming(false);
			};

			socket.onerror = (error) => {
				console.error("WebSocket error:", error);
				setIsStreaming(false);
			};
		} catch (error) {
			console.error("Error starting stream:", error);
			setIsStreaming(false);
		}
	};

	const stopStreaming = () => {
		console.log("Stopping streaming...");
		setIsStreaming(false);

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
		if (socketRef.current) {
			socketRef.current.close();
			socketRef.current = null;
		}

		audioBufferRef.current = [];
		accumulatedSamplesRef.current = 0;
		console.log("Streaming stopped");
	};

	return (
		<div>
			<h2>🎙️ Audio Streaming Test</h2>
			<button onClick={isStreaming ? stopStreaming : startStreaming}>{isStreaming ? "Stop Streaming" : "Start Streaming"}</button>
			{isStreaming && <p>Streaming audio in {chunkDuration} second chunks...</p>}
			{isStreaming && <p>Listening for server playback...</p>}
		</div>
	);
};

export default AudioStreamer;