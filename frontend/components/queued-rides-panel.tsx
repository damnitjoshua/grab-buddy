"use client"

import type { QueuedRide } from "@/lib/types"
import { Bell, X, Check, XIcon } from "lucide-react"

interface QueuedRidesPanelProps {
  queuedRides: QueuedRide[]
  onViewRide: (index: number) => void
  onAccept: (index: number) => void
  onDecline: (index: number) => void
  onClose: () => void
  isOpen: boolean
}

export default function QueuedRidesPanel({
  queuedRides,
  onViewRide,
  onAccept,
  onDecline,
  onClose,
  isOpen,
}: QueuedRidesPanelProps) {
  if (!isOpen) return null

  // Separate accepted and pending rides
  const acceptedRides = queuedRides.filter((ride) => ride.status === "accepted")
  const pendingRides = queuedRides.filter((ride) => ride.status === "pending")

  // Sort rides: accepted first, then pending
  const sortedRides = [...acceptedRides, ...pendingRides]

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-lg w-full max-w-md max-h-[80vh] flex flex-col">
        <div className="bg-green-500 text-white p-3 flex items-center justify-between rounded-t-lg">
          <div className="flex items-center">
            <Bell size={18} className="mr-2" />
            <span className="font-medium">Queued Ride Requests</span>
          </div>
          <button onClick={onClose} className="p-1 rounded-full hover:bg-green-600">
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          {queuedRides.length === 0 ? (
            <div className="p-6 text-center text-gray-500">
              <p>No queued ride requests</p>
            </div>
          ) : (
            <div className="divide-y">
              {sortedRides.map((ride, index) => {
                // Find the original index in the queuedRides array
                const originalIndex = queuedRides.findIndex((r) => r.id === ride.id)

                return (
                  <div
                    key={ride.id}
                    className={`p-4 hover:bg-gray-50 ${ride.status === "accepted" ? "bg-green-50" : ""}`}
                  >
                    <div className="flex justify-between items-center mb-1">
                      <div className="flex items-center">
                        <div className="w-8 h-8 rounded-full bg-gray-200 mr-2 overflow-hidden">
                          <img
                            src="/placeholder.svg?height=32&width=32"
                            alt="Passenger"
                            className="w-full h-full object-cover"
                          />
                        </div>
                        <span className="font-medium">{ride.passengerName}</span>
                      </div>
                      <div className="flex items-center">
                        <span
                          className={`font-medium ${ride.status === "accepted" ? "text-green-500" : "text-gray-700"}`}
                        >
                          ${ride.fare.toFixed(2)}
                        </span>
                        {ride.status === "accepted" && (
                          <span className="ml-2 px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">
                            Accepted
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="ml-10">
                      <div className="text-xs text-gray-500 mb-1">
                        <span className="font-medium">Pickup:</span> {ride.pickupLocation.address}
                      </div>
                      <div className="text-xs text-gray-500">
                        <span className="font-medium">Dropoff:</span> {ride.dropoffLocation.address}
                      </div>
                      <div className="flex items-center mt-1 text-xs text-gray-500">
                        <span className="mr-3">{ride.distance} km</span>
                        <span>{ride.estimatedTime} min</span>
                      </div>
                    </div>

                    {/*  min</span>
                      </div>
                    </div>

                    {/* Accept/Decline buttons - only show for pending rides */}
                    {ride.status !== "accepted" && (
                      <div className="flex mt-3 gap-2">
                        <button
                          onClick={() => onAccept(originalIndex)}
                          className="flex-1 py-1 rounded-lg bg-green-500 text-white font-medium flex items-center justify-center"
                        >
                          <Check size={16} className="mr-1" />
                          Accept
                        </button>
                        <button
                          onClick={() => onDecline(originalIndex)}
                          className="flex-1 py-1 rounded-lg border border-red-300 text-red-500 font-medium flex items-center justify-center"
                        >
                          <XIcon size={16} className="mr-1" />
                          Decline
                        </button>
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>

        <div className="p-3 border-t">
          <button onClick={onClose} className="w-full py-2 rounded-lg bg-gray-200 font-medium">
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
