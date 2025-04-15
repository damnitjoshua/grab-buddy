"use client"

import { useState, useEffect, useRef } from "react"
import { DriverStatus, RideStatus, type Ride, type Message, type QueuedRide } from "./types"
import { generateMockRide } from "./mock-data"

export function useDriverState() {
  const [driverStatus, setDriverStatus] = useState<DriverStatus>(DriverStatus.OFFLINE)
  const [rideStatus, setRideStatus] = useState<RideStatus>(RideStatus.IDLE)
  const [currentRide, setCurrentRide] = useState<Ride | null>(null)
  const [nextRide, setNextRide] = useState<Ride | null>(null) // For the confirmation modal
  const [requestTimer, setRequestTimer] = useState<NodeJS.Timeout | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [rideRequests, setRideRequests] = useState<QueuedRide[]>([])
  const [currentRequestIndex, setCurrentRequestIndex] = useState(0)
  const [queuedRides, setQueuedRides] = useState<QueuedRide[]>([])
  const [showNewRequestPopup, setShowNewRequestPopup] = useState(false)
  const [showQueuedRidesPanel, setShowQueuedRidesPanel] = useState(false)
  const [lastRequestTime, setLastRequestTime] = useState(0) // Track when the last request was generated
  const [activeRideRequestTimer, setActiveRideRequestTimer] = useState<NodeJS.Timeout | null>(null) // Separate timer for active ride requests
  const [showPendingRequestsAfterRide, setShowPendingRequestsAfterRide] = useState(false) // New state for showing pending requests after ride

  // Maximum number of pending requests allowed
  const MAX_PENDING_REQUESTS = 3

  // Reference to speech synthesis
  const speechSynthesisRef = useRef<SpeechSynthesis | null>(null)

  // Initialize speech synthesis
  useEffect(() => {
    if (typeof window !== "undefined") {
      speechSynthesisRef.current = window.speechSynthesis
    }

    // Clean up on unmount
    return () => {
      if (speechSynthesisRef.current && speechSynthesisRef.current.speaking) {
        speechSynthesisRef.current.cancel()
      }
    }
  }, [])

  // Count pending requests in the queue
  const countPendingRequests = () => {
    return queuedRides.filter((ride) => ride.status === "pending").length
  }

  // Speak text using TTS
  const speak = (text: string) => {
    if (speechSynthesisRef.current) {
      // Cancel any ongoing speech
      if (speechSynthesisRef.current.speaking) {
        speechSynthesisRef.current.cancel()
      }

      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 1.0
      utterance.pitch = 1.0
      utterance.volume = 1.0
      speechSynthesisRef.current.speak(utterance)
    }
  }

  // Stop any ongoing speech
  const stopSpeech = () => {
    if (speechSynthesisRef.current && speechSynthesisRef.current.speaking) {
      speechSynthesisRef.current.cancel()
    }
  }

  // Generate a new ride request specifically for active rides
  const generateActiveRideRequest = () => {
    if (
      driverStatus === DriverStatus.ONLINE &&
      (rideStatus === RideStatus.ACCEPTED ||
        rideStatus === RideStatus.PICKUP_REACHED ||
        rideStatus === RideStatus.IN_PROGRESS)
    ) {
      // Only generate a new request if we're under the maximum pending requests
      if (countPendingRequests() < MAX_PENDING_REQUESTS) {
        const newRide = generateMockRide()
        setLastRequestTime(Date.now())

        setQueuedRides((prev) => {
          // Add the new ride to queued rides
          const updatedQueued = [...prev, { ...newRide, status: "pending", timestamp: new Date() }]

          // Show notification popup
          setShowNewRequestPopup(true)
          setTimeout(() => setShowNewRequestPopup(false), 5000)

          // Announce new ride request with TTS
          speak("New ride request received")

          return updatedQueued
        })
      }

      // Schedule the next active ride request with longer gaps
      const nextTimer = setTimeout(generateActiveRideRequest, 15000 + Math.random() * 15000) // 15-30 seconds
      setActiveRideRequestTimer(nextTimer)
    }
  }

  // Generate a new ride request
  const generateNewRideRequest = () => {
    const now = Date.now()

    // If we're IDLE or in REQUESTED state, add to the main requests queue
    if (rideStatus === RideStatus.IDLE || rideStatus === RideStatus.REQUESTED) {
      // Only generate if we don't already have pending requests
      if (rideRequests.length === 0) {
        const newRide = generateMockRide()
        setLastRequestTime(now)

        setRideRequests((prev) => {
          const updatedRequests = [...prev, { ...newRide, status: "pending", timestamp: new Date() }]

          // If this is the first request and we're idle, set the ride status to REQUESTED
          if (rideStatus === RideStatus.IDLE && updatedRequests.length === 1) {
            setRideStatus(RideStatus.REQUESTED)
            setCurrentRide(newRide)

            // Announce the new ride with TTS
            const ttsText = `New ride request from ${newRide.passengerName}. Pickup at ${newRide.pickupLocation.address}. Estimated fare: ${newRide.fare.toFixed(2)} dollars.`
            speak(ttsText)
          }

          return updatedRequests
        })
      }
    }
    // If we're in an active ride, add to queued rides
    else if (
      rideStatus === RideStatus.ACCEPTED ||
      rideStatus === RideStatus.PICKUP_REACHED ||
      rideStatus === RideStatus.IN_PROGRESS
    ) {
      // Only generate a new request if we're under the maximum pending requests
      if (countPendingRequests() < MAX_PENDING_REQUESTS) {
        const newRide = generateMockRide()
        setLastRequestTime(now)

        setQueuedRides((prev) => {
          // Add the new ride to queued rides
          const updatedQueued = [...prev, { ...newRide, status: "pending", timestamp: new Date() }]

          // Show notification popup
          setShowNewRequestPopup(true)
          setTimeout(() => setShowNewRequestPopup(false), 5000)

          // Announce new ride request with TTS
          speak("New ride request received")

          return updatedQueued
        })
      }
    }

    // Schedule the next request if still online - use longer time for less frequent requests
    if (driverStatus === DriverStatus.ONLINE) {
      const timer = setTimeout(generateNewRideRequest, 10000 + Math.random() * 20000) // 10-30 seconds
      setRequestTimer(timer)
    }
  }

  // Start active ride request generation when entering an active ride state
  useEffect(() => {
    // When entering an active ride state, start generating active ride requests
    if (
      driverStatus === DriverStatus.ONLINE &&
      (rideStatus === RideStatus.ACCEPTED ||
        rideStatus === RideStatus.PICKUP_REACHED ||
        rideStatus === RideStatus.IN_PROGRESS)
    ) {
      // Clear any existing active ride request timer
      if (activeRideRequestTimer) {
        clearTimeout(activeRideRequestTimer)
      }

      // Start generating active ride requests with a delay
      const timer = setTimeout(generateActiveRideRequest, 8000 + Math.random() * 7000) // 8-15 seconds
      setActiveRideRequestTimer(timer)
    }

    // Clean up when leaving active ride states
    return () => {
      if (activeRideRequestTimer) {
        clearTimeout(activeRideRequestTimer)
        setActiveRideRequestTimer(null)
      }
    }
  }, [rideStatus, driverStatus])

  // Toggle driver online/offline status
  const toggleDriverStatus = () => {
    if (driverStatus === DriverStatus.OFFLINE) {
      setDriverStatus(DriverStatus.ONLINE)
      setRideStatus(RideStatus.IDLE)

      const newRide = generateMockRide()
      setRideRequests([{ ...newRide, status: "pending", timestamp: new Date() }])
      setCurrentRide(newRide)
      setRideStatus(RideStatus.REQUESTED)

      // Announce the new ride with TTS
      const ttsText = `New ride request from ${newRide.passengerName}. Pickup at ${newRide.pickupLocation.address}. Estimated fare: ${newRide.fare.toFixed(2)} dollars.`
      speak(ttsText)

      // Schedule next requests with longer delay
      const nextTimer = setTimeout(generateNewRideRequest, 10000 + Math.random() * 10000) // 10-20 seconds
      setRequestTimer(nextTimer)

      setRequestTimer(timer)
    } else {
      setDriverStatus(DriverStatus.OFFLINE)
      if (requestTimer) {
        clearTimeout(requestTimer)
        setRequestTimer(null)
      }
      if (activeRideRequestTimer) {
        clearTimeout(activeRideRequestTimer)
        setActiveRideRequestTimer(null)
      }
      // Only reset if not in an active ride
      if (rideStatus === RideStatus.IDLE || rideStatus === RideStatus.REQUESTED) {
        setRideStatus(RideStatus.IDLE)
        setCurrentRide(null)
        setRideRequests([])
        setCurrentRequestIndex(0)
      }
    }
  }

  // Toggle queued rides panel
  const toggleQueuedRidesPanel = () => {
    setShowQueuedRidesPanel((prev) => !prev)

    // When opening the panel, hide the notification popup
    if (!showQueuedRidesPanel) {
      setShowNewRequestPopup(false)
    }
  }

  // Close pending requests panel after ride
  const closePendingRequestsPanel = () => {
    setShowPendingRequestsAfterRide(false)
    goBackToIdle()
  }

  // Add a message to the conversation
  const addMessage = (text: string, isDriver = true) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      text,
      isDriver,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, newMessage])
  }

  // Navigate to the next ride request
  const nextRequest = () => {
    if (rideRequests.length > 1) {
      const newIndex = (currentRequestIndex + 1) % rideRequests.length
      setCurrentRequestIndex(newIndex)
      setCurrentRide(rideRequests[newIndex])
    }
  }

  // Navigate to the previous ride request
  const prevRequest = () => {
    if (rideRequests.length > 1) {
      const newIndex = (currentRequestIndex - 1 + rideRequests.length) % rideRequests.length
      setCurrentRequestIndex(newIndex)
      setCurrentRide(rideRequests[newIndex])
    }
  }

  // Accept current ride request
  const acceptRide = () => {
    // Stop any ongoing TTS
    stopSpeech()

    if (currentRide && rideRequests.length > 0) {
      // Update the status of the current request to accepted
      setRideRequests((prev) =>
        prev.map((req, index) => (index === currentRequestIndex ? { ...req, status: "accepted" } : req)),
      )

      // Move other pending requests to the queued rides
      const otherPendingRequests = rideRequests.filter(
        (_, index) => index !== currentRequestIndex && _.status === "pending",
      )

      setQueuedRides((prev) => [...prev, ...otherPendingRequests])

      // Clear pending requests except the accepted one
      setRideRequests((prev) => prev.filter((_, index) => index === currentRequestIndex))
      setCurrentRequestIndex(0)

      // Update ride status
      setRideStatus(RideStatus.ACCEPTED)

      // Auto-send a message when accepting the ride
      addMessage("Hi! I've accepted your ride request and I'm on my way to pick you up.")
    }
  }

  // Accept a ride from the queued rides
  const acceptQueuedRide = (index: number) => {
    // Stop any ongoing TTS
    stopSpeech()

    if (queuedRides.length > index) {
      // If we're in the middle of a ride, just mark it as accepted but don't start yet
      if (
        rideStatus === RideStatus.ACCEPTED ||
        rideStatus === RideStatus.PICKUP_REACHED ||
        rideStatus === RideStatus.IN_PROGRESS
      ) {
        // Update the ride status to "accepted" in the queue
        setQueuedRides((prev) => prev.map((r, i) => (i === index ? { ...r, status: "accepted" } : r)))

        // Close the panel
        setShowQueuedRidesPanel(false)
        return
      }

      // If we're showing pending requests after a ride
      if (showPendingRequestsAfterRide) {
        const ride = queuedRides[index]
        setQueuedRides((prev) => prev.filter((_, i) => i !== index))
        setCurrentRide(ride)
        setRideStatus(RideStatus.ACCEPTED) // Go directly to ACCEPTED state
        setShowPendingRequestsAfterRide(false)

        // Auto-send a message when accepting the ride
        addMessage("Hi! I've accepted your ride request and I'm on my way to pick you up.")
        return
      }

      // Otherwise, we can start this ride immediately
      const ride = queuedRides[index]
      setQueuedRides((prev) => prev.filter((_, i) => i !== index))
      setCurrentRide(ride)
      setRideRequests([{ ...ride, status: "pending" }])
      setCurrentRequestIndex(0)
      setRideStatus(RideStatus.REQUESTED)

      // Close the panel
      setShowQueuedRidesPanel(false)
      // Hide the notification popup
      setShowNewRequestPopup(false)
    }
  }

  // Decline a ride from the queued rides
  const declineQueuedRide = (index: number) => {
    // Stop any ongoing TTS
    stopSpeech()

    if (queuedRides.length > index) {
      setQueuedRides((prev) => prev.filter((_, i) => i !== index))

      // If we're showing pending requests after a ride and there are no more pending requests
      if (showPendingRequestsAfterRide) {
        const pendingCount = queuedRides.filter((r) => r.status === "pending").length
        if (pendingCount <= 1) {
          // We're removing one, so check if there's 1 or fewer
          setShowPendingRequestsAfterRide(false)
          goBackToIdle()
        }
      }
    }
  }

  // Decline current ride request
  const declineRide = () => {
    // Stop any ongoing TTS
    stopSpeech()

    if (rideRequests.length > 0) {
      // Remove the current request
      setRideRequests((prev) => prev.filter((_, index) => index !== currentRequestIndex))

      // Adjust the current index if needed
      if (currentRequestIndex >= rideRequests.length - 1) {
        setCurrentRequestIndex(Math.max(0, rideRequests.length - 2))
      }

      // If no more requests, go back to IDLE
      if (rideRequests.length <= 1) {
        setRideStatus(RideStatus.IDLE)
        setCurrentRide(null)

        // If we're still online, schedule a new request
        if (driverStatus === DriverStatus.ONLINE) {
          const timer = setTimeout(generateNewRideRequest, 5000 + Math.random() * 5000) // 5-10 seconds
          setRequestTimer(timer)
        }
      } else {
        // Otherwise, set the current ride to the new current request
        const newIndex = Math.min(currentRequestIndex, rideRequests.length - 2)
        setCurrentRide(rideRequests[newIndex])
        setCurrentRequestIndex(newIndex)
      }
    }
  }

  // Decline all ride requests
  const declineAllRides = () => {
    // Stop any ongoing TTS
    stopSpeech()

    setRideRequests([])
    setCurrentRequestIndex(0)
    setRideStatus(RideStatus.IDLE)
    setCurrentRide(null)

    // If we're still online, schedule a new request
    if (driverStatus === DriverStatus.ONLINE) {
      const timer = setTimeout(generateNewRideRequest, 5000 + Math.random() * 5000) // 5-10 seconds
      setRequestTimer(timer)
    }
  }

  // Start ride (after reaching pickup)
  const startRide = () => {
    if (rideStatus === RideStatus.ACCEPTED) {
      setRideStatus(RideStatus.PICKUP_REACHED)
      // Auto-send a message when arriving at pickup
      addMessage("I've arrived at the pickup location. I'm waiting outside.")
    } else if (rideStatus === RideStatus.PICKUP_REACHED) {
      setRideStatus(RideStatus.IN_PROGRESS)
      // Auto-send a message when starting the ride
      addMessage(
        "We're on our way to your destination. Estimated arrival time is " + currentRide?.estimatedTime + " minutes.",
      )
    }
  }

  // End ride
  const endRide = () => {
    setRideStatus(RideStatus.COMPLETED)
    // Auto-send a message when ending the ride
    addMessage("We've arrived at your destination. Thank you for riding with Grab!")
  }

  // Accept the next ride after confirmation
  const acceptNextRide = () => {
    // Stop any ongoing TTS
    stopSpeech()

    if (nextRide) {
      setCurrentRide(nextRide)
      setRideStatus(RideStatus.ACCEPTED) // Set directly to ACCEPTED instead of REQUESTED
      setNextRide(null)

      // Auto-send a message when accepting the ride
      addMessage("Hi! I'm on my way to pick you up for your previously accepted ride.")
    }
  }

  // Decline the next ride after confirmation
  const declineNextRide = () => {
    // Stop any ongoing TTS
    stopSpeech()

    setNextRide(null)

    // Check if there are more accepted rides in the queue
    const acceptedRides = queuedRides.filter((ride) => ride.status === "accepted")

    if (acceptedRides.length > 0) {
      // Show the next accepted ride
      const nextAcceptedRide = acceptedRides[0]
      setNextRide(nextAcceptedRide)
      setQueuedRides((prev) => prev.filter((ride) => ride.id !== nextAcceptedRide.id))
    } else {
      // Check if there are pending rides
      const pendingRides = queuedRides.filter((ride) => ride.status === "pending")
      if (pendingRides.length > 0) {
        // Show pending rides panel
        setShowPendingRequestsAfterRide(true)
      } else {
        // No more accepted or pending rides, go back to IDLE and generate new request
        goBackToIdle()
      }
    }
  }

  // Reset after ride completion
  const resetRide = () => {
    // Reset ride-specific state
    setMessages([])

    // Check if there are any accepted rides in the queue
    const acceptedRides = queuedRides.filter((ride) => ride.status === "accepted")

    if (acceptedRides.length > 0) {
      // Show confirmation for the next accepted ride
      const nextAcceptedRide = acceptedRides[0]
      setNextRide(nextAcceptedRide)
      setQueuedRides((prev) => prev.filter((ride) => ride.id !== nextAcceptedRide.id))
      setRideStatus(RideStatus.CONFIRMING_NEXT_RIDE)
    } else {
      // Check if there are pending rides
      const pendingRides = queuedRides.filter((ride) => ride.status === "pending")
      if (pendingRides.length > 0) {
        // Show pending rides panel
        setShowPendingRequestsAfterRide(true)
      } else {
        // No accepted or pending rides, go back to IDLE state
        goBackToIdle()
      }
    }
  }

  // Helper function to go back to IDLE state and generate a new ride
  const goBackToIdle = () => {
    setRideStatus(RideStatus.IDLE)
    setCurrentRide(null)

    // Clear any existing timer
    if (requestTimer) {
      clearTimeout(requestTimer)
    }

    // Schedule a new ride request after a short delay
    const timer = setTimeout(() => {
      if (driverStatus === DriverStatus.ONLINE) {
        const newRide = generateMockRide()
        setRideRequests([{ ...newRide, status: "pending", timestamp: new Date() }])
        setCurrentRide(newRide)
        setRideStatus(RideStatus.REQUESTED)

        // Announce the new ride with TTS
        const ttsText = `New ride request from ${newRide.passengerName}. Pickup at ${newRide.pickupLocation.address}. Estimated fare: ${newRide.fare.toFixed(2)} dollars.`
        speak(ttsText)

        // Schedule next requests
        const nextTimer = setTimeout(generateNewRideRequest, 10000 + Math.random() * 20000) // 10-30 seconds
        setRequestTimer(nextTimer)
      }
    }, 3000) // 3-second delay

    setRequestTimer(timer)
  }

  // View a queued ride
  const viewQueuedRide = (index: number) => {
    if (queuedRides.length > index) {
      // If we're in the middle of a ride, just show the details
      if (
        rideStatus === RideStatus.ACCEPTED ||
        rideStatus === RideStatus.PICKUP_REACHED ||
        rideStatus === RideStatus.IN_PROGRESS
      ) {
        // Just mark it as viewed, don't start the ride yet
        return
      }

      // Otherwise, we can start this ride
      const ride = queuedRides[index]
      setQueuedRides((prev) => prev.filter((_, i) => i !== index))
      setCurrentRide(ride)
      setRideRequests([{ ...ride, status: "pending" }])
      setCurrentRequestIndex(0)
      setRideStatus(RideStatus.REQUESTED)
    }
  }

  // Send a custom message
  const sendMessage = (text: string) => {
    if (text.trim()) {
      addMessage(text)
    }
  }

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (requestTimer) {
        clearTimeout(requestTimer)
      }
      if (activeRideRequestTimer) {
        clearTimeout(activeRideRequestTimer)
      }
    }
  }, [requestTimer, activeRideRequestTimer])

  // Auto-expire old requests
  useEffect(() => {
    const interval = setInterval(() => {
      const now = new Date()
      setRideRequests((prev) => {
        const updated = prev.map((req) => {
          // Expire requests older than 30 seconds
          if (req.status === "pending" && now.getTime() - req.timestamp.getTime() > 30000) {
            return { ...req, status: "expired" }
          }
          return req
        })

        // Remove expired requests
        return updated.filter((req) => req.status !== "expired")
      })
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  // Add this useEffect near the other useEffects
  useEffect(() => {
    // Only run this if we're IDLE and ONLINE but have no pending requests
    if (driverStatus === DriverStatus.ONLINE && rideStatus === RideStatus.IDLE && rideRequests.length === 0) {
      const now = Date.now()
      if (now - lastRequestTime > 5000) {
        const timer = setTimeout(() => {
          const newRide = generateMockRide()
          setRideRequests([{ ...newRide, status: "pending", timestamp: new Date() }])
          setCurrentRide(newRide)
          setRideStatus(RideStatus.REQUESTED)

          // Announce the new ride with TTS
          const ttsText = `New ride request from ${newRide.passengerName}. Pickup at ${newRide.pickupLocation.address}. Estimated fare: ${newRide.fare.toFixed(2)} dollars.`
          speak(ttsText)

          setLastRequestTime(now)
        }, 2000)

        return () => clearTimeout(timer)
      }
    }
  }, [driverStatus, rideStatus, rideRequests.length, lastRequestTime])

  // Force generate a new ride request periodically but respect the MAX_PENDING_REQUESTS limit
  useEffect(() => {
    if (driverStatus === DriverStatus.ONLINE) {
      const interval = setInterval(() => {
        const now = Date.now()
        // Only generate if it's been more than 15 seconds since the last request
        // and we're under the maximum pending requests limit
        if (now - lastRequestTime > 15000 && countPendingRequests() < MAX_PENDING_REQUESTS) {
          if (
            rideStatus === RideStatus.ACCEPTED ||
            rideStatus === RideStatus.PICKUP_REACHED ||
            rideStatus === RideStatus.IN_PROGRESS
          ) {
            // Generate a new ride request for active rides
            const newRide = generateMockRide()
            setQueuedRides((prev) => {
              const updatedQueued = [...prev, { ...newRide, status: "pending", timestamp: new Date() }]

              // Show notification popup
              setShowNewRequestPopup(true)
              setTimeout(() => setShowNewRequestPopup(false), 5000)

              // Announce new ride request with TTS
              speak("New ride request received")

              return updatedQueued
            })
          }
          setLastRequestTime(now)
        }
      }, 30000) // Check every 30 seconds

      return () => clearInterval(interval)
    }
  }, [driverStatus, rideStatus, lastRequestTime])

  // Count of pending (unattended) ride requests
  const pendingQueuedRidesCount = queuedRides.filter((ride) => ride.status === "pending").length

  return {
    driverStatus,
    rideStatus,
    currentRide,
    nextRide,
    messages,
    rideRequests,
    currentRequestIndex,
    queuedRides,
    pendingQueuedRidesCount,
    showNewRequestPopup,
    showQueuedRidesPanel,
    showPendingRequestsAfterRide,
    toggleDriverStatus,
    toggleQueuedRidesPanel,
    closePendingRequestsPanel,
    acceptRide,
    declineRide,
    declineAllRides,
    acceptQueuedRide,
    declineQueuedRide,
    startRide,
    endRide,
    resetRide,
    acceptNextRide,
    declineNextRide,
    sendMessage,
    nextRequest,
    prevRequest,
    viewQueuedRide,
  }
}
