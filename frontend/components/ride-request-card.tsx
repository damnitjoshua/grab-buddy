"use client"
import { useState, useEffect, useRef } from "react"
import type React from "react"

import type { QueuedRide } from "@/lib/types"
import { Navigation, DollarSign, Clock, ChevronLeft, ChevronRight } from "lucide-react"

interface RideRequestCardProps {
  rides: QueuedRide[]
  currentIndex: number
  onAccept: () => void
  onDecline: () => void
  onDeclineAll: () => void
  onNext: () => void
  onPrev: () => void
}

export default function RideRequestCard({
  rides,
  currentIndex,
  onAccept,
  onDecline,
  onDeclineAll,
  onNext,
  onPrev,
}: RideRequestCardProps) {
  if (!rides.length) return null

  const ride = rides[currentIndex]
  const [progressStyle, setProgressStyle] = useState<React.CSSProperties>({})
  const progressRef = useRef<HTMLDivElement>(null)
  const rideIdRef = useRef<string | null>(null)

  useEffect(() => {
    // Reset animation when switching between rides
    if (rideIdRef.current !== ride.id) {
      rideIdRef.current = ride.id

      const requestAge = new Date().getTime() - ride.timestamp.getTime()
      const isOld = requestAge > 20000 // 20 seconds
      const remainingTime = Math.max(0, 30000 - requestAge)
      const initialProgress = Math.min(100, (requestAge / 30000) * 100)

      setProgressStyle({
        width: `${initialProgress}%`,
        animation: `progress ${remainingTime / 1000}s linear forwards`,
      })
    }
  }, [ride.id, ride.timestamp])

  return (
    <div className="bg-white rounded-t-2xl shadow-lg p-4 animate-slide-up">
      <div className="flex justify-between items-center mb-2">
        <h2 className="text-lg font-bold">New Ride Request</h2>
        <div className="flex items-center">
          {rides.length > 1 && (
            <div className="flex items-center mr-2 text-sm">
              <span>{currentIndex + 1}</span>
              <span className="mx-1">/</span>
              <span>{rides.length}</span>
            </div>
          )}
        </div>
      </div>

      {/* Request expiration progress bar */}
      <div className="w-full h-1 bg-gray-200 rounded-full mb-3">
        <div
          ref={progressRef}
          className={`h-1 rounded-full ${new Date().getTime() - ride.timestamp.getTime() > 20000 ? "bg-red-500" : "bg-green-500"}`}
          style={progressStyle}
        ></div>
      </div>

      <div className="mb-4">
        <div className="flex items-center mb-1">
          <div className="w-8 h-8 rounded-full bg-gray-200 mr-2 overflow-hidden">
            <img src="/placeholder.svg?height=32&width=32" alt="Passenger" className="w-full h-full object-cover" />
          </div>
          <span className="font-medium">{ride.passengerName}</span>
        </div>
        <div className="flex items-center">
          <span className="text-yellow-500 mr-1">★</span>
          <span className="text-sm">{ride.passengerRating}</span>
        </div>
      </div>

      <div className="border-t border-b border-gray-200 py-3 mb-4">
        <div className="flex mb-3">
          <div className="mr-3 pt-1">
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <div className="w-0.5 h-12 bg-gray-300 mx-auto my-1"></div>
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
          </div>
          <div className="flex-1">
            <div className="mb-3">
              <p className="text-xs text-gray-500">PICKUP</p>
              <p className="font-medium">{ride.pickupLocation.address}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500">DROPOFF</p>
              <p className="font-medium">{ride.dropoffLocation.address}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 mb-4">
        <div className="flex flex-col items-center justify-center bg-gray-100 rounded-lg p-2">
          <Navigation size={18} className="mb-1 text-gray-700" />
          <span className="text-xs font-medium">{ride.distance} km</span>
        </div>
        <div className="flex flex-col items-center justify-center bg-gray-100 rounded-lg p-2">
          <DollarSign size={18} className="mb-1 text-gray-700" />
          <span className="text-xs font-medium">${ride.fare.toFixed(2)}</span>
        </div>
        <div className="flex flex-col items-center justify-center bg-gray-100 rounded-lg p-2">
          <Clock size={18} className="mb-1 text-gray-700" />
          <span className="text-xs font-medium">{ride.estimatedTime} min</span>
        </div>
      </div>

      <div className="flex gap-3">
        {rides.length > 1 && (
          <button onClick={onPrev} className="p-3 rounded-lg border border-gray-300" aria-label="Previous request">
            <ChevronLeft size={20} />
          </button>
        )}

        <button onClick={onDecline} className="flex-1 py-3 rounded-lg border border-gray-300 font-medium">
          Decline
        </button>

        <button onClick={onAccept} className="flex-1 py-3 rounded-lg bg-green-500 text-white font-medium">
          Accept
        </button>

        {rides.length > 1 && (
          <button onClick={onNext} className="p-3 rounded-lg border border-gray-300" aria-label="Next request">
            <ChevronRight size={20} />
          </button>
        )}
      </div>

      {rides.length > 1 && (
        <button
          onClick={onDeclineAll}
          className="w-full mt-3 py-2 rounded-lg border border-red-300 text-red-500 font-medium"
        >
          Decline All Requests
        </button>
      )}
    </div>
  )
}
