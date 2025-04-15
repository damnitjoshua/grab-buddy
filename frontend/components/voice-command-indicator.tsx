"use client"

import { Mic, MicOff } from "lucide-react"
import { useEffect, useState } from "react"

interface VoiceCommandIndicatorProps {
  isListeningForWakeWord: boolean
  isVoiceCommandActive: boolean
  lastCommand: string
  error: string | null
  onActivate: () => void
}

export default function VoiceCommandIndicator({
  isListeningForWakeWord,
  isVoiceCommandActive,
  lastCommand,
  error,
  onActivate,
}: VoiceCommandIndicatorProps) {
  const [showLastCommand, setShowLastCommand] = useState(false)

  // Show the last command briefly when it changes
  useEffect(() => {
    if (lastCommand) {
      setShowLastCommand(true)
      const timer = setTimeout(() => {
        setShowLastCommand(false)
      }, 3000) // Hide after 3 seconds

      return () => clearTimeout(timer)
    }
  }, [lastCommand])

  return (
    <div className="absolute top-16 left-4 z-30">
      <div className="flex flex-col items-start">
        {/* Voice status indicator */}
        <button
          onClick={onActivate}
          className={`flex items-center gap-2 rounded-full px-3 py-2 shadow-lg ${
            isVoiceCommandActive
              ? "bg-green-500 text-white animate-pulse"
              : isListeningForWakeWord
                ? "bg-white text-gray-700 border border-gray-200"
                : "bg-gray-200 text-gray-500"
          }`}
        >
          {isVoiceCommandActive ? (
            <>
              <Mic size={16} />
              <span className="text-sm font-medium">Voice Commands Active</span>
            </>
          ) : isListeningForWakeWord ? (
            <>
              <Mic size={16} />
              <span className="text-sm font-medium">Say "Hey Grab"</span>
            </>
          ) : (
            <>
              <MicOff size={16} />
              <span className="text-sm font-medium">Voice inactive</span>
            </>
          )}
        </button>

        {/* Last command display */}

        {/* Error message */}
        {error && (
          <div className="mt-2 bg-red-100 text-red-800 rounded-lg shadow-lg p-2 text-sm max-w-xs">
            <p>{error}</p>
          </div>
        )}

        {/* Help text for active voice commands */}
      </div>
    </div>
  )
}
