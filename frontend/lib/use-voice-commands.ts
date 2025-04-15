"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { DriverStatus, type RideStatus } from "./types"

// Define the available voice commands
type VoiceCommand =
  | "go online"
  | "go offline"
  | "accept ride"
  | "decline ride"
  | "arrived at pickup"
  | "start ride"
  | "end ride"
  | "view requests"
  | "cancel"

// Define the props for the hook
interface UseVoiceCommandsProps {
  driverStatus: DriverStatus
  rideStatus: RideStatus
  toggleDriverStatus: () => void
  acceptRide: () => void
  declineRide: () => void
  startRide: () => void
  endRide: () => void
  toggleQueuedRidesPanel: () => void
}

// Declare SpeechRecognition
declare var SpeechRecognition: any
declare var webkitSpeechRecognition: any

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
  // State for tracking if voice recognition is supported
  const [isSupported, setIsSupported] = useState(false)

  // State for tracking if we're listening for the wake word
  const [isListeningForWakeWord, setIsListeningForWakeWord] = useState(false)

  // State for tracking if voice commands are active (after wake word detected)
  const [isVoiceCommandActive, setIsVoiceCommandActive] = useState(false)

  // State for tracking the last recognized command
  const [lastCommand, setLastCommand] = useState<string>("")

  // State for tracking error messages
  const [error, setError] = useState<string | null>(null)

  // Refs for the speech recognition instances
  const wakeWordRecognitionRef = useRef<SpeechRecognition | null>(null)
  const commandRecognitionRef = useRef<SpeechRecognition | null>(null)

  // Initialize speech recognition
  useEffect(() => {
    // Check if the browser supports the Web Speech API
    if (typeof window !== "undefined" && ("SpeechRecognition" in window || "webkitSpeechRecognition" in window)) {
      setIsSupported(true)

      // Create the SpeechRecognition instances
      const SpeechRecognition = window.SpeechRecognition || webkitSpeechRecognition

      // Initialize wake word recognition
      wakeWordRecognitionRef.current = new SpeechRecognition()
      wakeWordRecognitionRef.current.continuous = true
      wakeWordRecognitionRef.current.interimResults = false

      // Initialize command recognition
      commandRecognitionRef.current = new SpeechRecognition()
      commandRecognitionRef.current.continuous = false
      commandRecognitionRef.current.interimResults = false

      // Start listening for wake word
      startWakeWordDetection()
    } else {
      setError("Speech recognition is not supported in this browser")
    }

    // Cleanup function
    return () => {
      stopWakeWordDetection()
      stopCommandRecognition()
    }
  }, [])

  // Function to start listening for the wake word
  const startWakeWordDetection = useCallback(() => {
    if (!wakeWordRecognitionRef.current) return

    try {
      wakeWordRecognitionRef.current.onresult = (event) => {
        const transcript = event.results[event.results.length - 1][0].transcript.toLowerCase().trim()
        console.log("Wake word detection heard:", transcript)

        // Check for wake word "hey grab"
        if (transcript.includes("hey grab")) {
          console.log("Wake word detected!")

          // Stop wake word detection and start command recognition
          stopWakeWordDetection()
          startCommandRecognition()

          // Provide audio feedback
          speakFeedback("Voice commands activated. How can I help?")
        }
      }

      wakeWordRecognitionRef.current.onend = () => {
        // Restart wake word detection if it ends and we're still in wake word mode
        if (isListeningForWakeWord) {
          console.log("Wake word detection ended, restarting...")
          wakeWordRecognitionRef.current?.start()
        }
      }

      wakeWordRecognitionRef.current.onerror = (event) => {
        console.error("Wake word detection error:", event.error)
        // Don't set error state for "no-speech" errors as they're common
        if (event.error !== "no-speech") {
          setError(`Wake word detection error: ${event.error}`)
        }

        // Restart wake word detection on error
        if (isListeningForWakeWord) {
          setTimeout(() => {
            wakeWordRecognitionRef.current?.start()
          }, 100)
        }
      }

      wakeWordRecognitionRef.current.start()
      setIsListeningForWakeWord(true)
      setError(null)
    } catch (err) {
      console.error("Error starting wake word detection:", err)
      setError("Failed to start wake word detection")
    }
  }, [isListeningForWakeWord])

  // Function to stop wake word detection
  const stopWakeWordDetection = useCallback(() => {
    if (!wakeWordRecognitionRef.current) return

    try {
      wakeWordRecognitionRef.current.stop()
      setIsListeningForWakeWord(false)
    } catch (err) {
      console.error("Error stopping wake word detection:", err)
    }
  }, [])

  // Function to start command recognition
  const startCommandRecognition = useCallback(() => {
    if (!commandRecognitionRef.current) return

    try {
      commandRecognitionRef.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript.toLowerCase().trim()
        console.log("Command recognition heard:", transcript)
        setLastCommand(transcript)

        // Process the command
        processCommand(transcript)
      }

      commandRecognitionRef.current.onend = () => {
        // If voice commands are still active, restart command recognition
        if (isVoiceCommandActive) {
          console.log("Command recognition ended, restarting...")
          setTimeout(() => {
            commandRecognitionRef.current?.start()
          }, 100)
        }
      }

      commandRecognitionRef.current.onerror = (event) => {
        console.error("Command recognition error:", event.error)
        if (event.error !== "no-speech") {
          setError(`Command recognition error: ${event.error}`)
        }

        // Restart command recognition on error if still active
        if (isVoiceCommandActive) {
          setTimeout(() => {
            commandRecognitionRef.current?.start()
          }, 100)
        }
      }

      commandRecognitionRef.current.start()
      setIsVoiceCommandActive(true)
    } catch (err) {
      console.error("Error starting command recognition:", err)
      setError("Failed to start command recognition")
    }
  }, [isVoiceCommandActive])

  // Function to stop command recognition
  const stopCommandRecognition = useCallback(() => {
    if (!commandRecognitionRef.current) return

    try {
      commandRecognitionRef.current.stop()
      setIsVoiceCommandActive(false)
    } catch (err) {
      console.error("Error stopping command recognition:", err)
    }
  }, [])

  // Function to process voice commands - SIMPLIFIED to just online/offline
  const processCommand = useCallback(
    (command: string) => {
      console.log("Processing command:", command)

      // Handle deactivation commands
      if (
        command.includes("bye grab") ||
        command.includes("buy grab") ||
        command.includes("goodbye grab") ||
        command.includes("stop listening") ||
        command.includes("turn off voice commands")
      ) {
        speakFeedback("Voice commands deactivated")
        stopCommandRecognition()
        startWakeWordDetection()
        return
      }

      // Handle driver status commands - ONLY THESE COMMANDS ARE KEPT
      if (command.includes("go online")) {
        if (driverStatus === DriverStatus.OFFLINE) {
          toggleDriverStatus()
          speakFeedback("Going online")
        } else {
          speakFeedback("You are already online")
        }
      } else if (command.includes("go offline")) {
        if (driverStatus === DriverStatus.ONLINE) {
          toggleDriverStatus()
          speakFeedback("Going offline")
        } else {
          speakFeedback("You are already offline")
        }
      }
      // Handle help command
      else if (command.includes("help") || command.includes("what can you do")) {
        speakFeedback("Available commands include: go online and go offline")
      }
      // Unknown command
      else {
        speakFeedback(
          "Sorry, I didn't understand that command. You can say go online, go offline, or bye grab to exit.",
        )
      }
    },
    [driverStatus, toggleDriverStatus, stopCommandRecognition, startWakeWordDetection],
  )

  // Function to provide audio feedback
  const speakFeedback = useCallback((text: string) => {
    if ("speechSynthesis" in window) {
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 1.0
      utterance.pitch = 1.0
      utterance.volume = 1.0
      window.speechSynthesis.speak(utterance)
    }
  }, [])

  // Function to manually activate voice commands (for testing)
  const activateVoiceCommands = useCallback(() => {
    if (isVoiceCommandActive) {
      // If already active, deactivate
      speakFeedback("Voice commands deactivated")
      stopCommandRecognition()
      startWakeWordDetection()
    } else {
      // If not active, activate
      stopWakeWordDetection()
      startCommandRecognition()
      speakFeedback("Voice commands activated. How can I help?")
    }
  }, [
    isVoiceCommandActive,
    stopWakeWordDetection,
    startCommandRecognition,
    stopCommandRecognition,
    startWakeWordDetection,
    speakFeedback,
  ])

  return {
    isSupported,
    isListeningForWakeWord,
    isVoiceCommandActive,
    lastCommand,
    error,
    activateVoiceCommands,
  }
}
