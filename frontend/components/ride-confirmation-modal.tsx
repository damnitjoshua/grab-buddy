"use client"

import type { Ride } from "@/lib/types"
import { CheckCircle, X } from "lucide-react"

interface RideConfirmationModalProps {
  ride: Ride
  onAccept: () => void
  onDecline: () => void
}

export default function RideConfirmationModal({ ride, onAccept, onDecline }: RideConfirmationModalProps) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-lg w-full max-w-md">
        <div className="p-4 border-b">
          <div className="flex justify-between items-center">
            <h2 className="text-lg font-bold">Start Previously Accepted Ride</h2>
            <button onClick={onDecline} className="p-1 rounded-full hover:bg-gray-100">
              <X size={20} />
            </button>
          </div>
        </div>

        <div className="p-4">
          <div className="flex items-center mb-4">
            <div className="w-10 h-10 rounded-full bg-gray-200 overflow-hidden mr-3">
              <img src="/placeholder.svg?height=40&width=40" alt="Passenger" className="w-full h-full object-cover" />
            </div>
            <div>
              <p className="font-medium">{ride.passengerName}</p>
              <div className="flex items-center">
                <span className="text-yellow-500 mr-1">★</span>
                <span className="text-sm">{ride.passengerRating}</span>
              </div>
            </div>
          </div>

          <div className="border rounded-lg p-3 mb-4">
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

            <div className="grid grid-cols-3 gap-2 pt-2 border-t">
              <div className="text-center">
                <p className="text-xs text-gray-500">DISTANCE</p>
                <p className="font-medium">{ride.distance} km</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-500">FARE</p>
                <p className="font-medium">${ride.fare.toFixed(2)}</p>
              </div>
              <div className="text-center">
                <p className="text-xs text-gray-500">TIME</p>
                <p className="font-medium">{ride.estimatedTime} min</p>
              </div>
            </div>
          </div>

          <p className="text-center mb-4">Your previous ride is complete. Do you want to start this ride now?</p>

          <div className="flex gap-3">
            <button onClick={onDecline} className="flex-1 py-3 rounded-lg border border-gray-300 font-medium">
              Decline
            </button>
            <button
              onClick={onAccept}
              className="flex-1 py-3 rounded-lg bg-green-500 text-white font-medium flex items-center justify-center"
            >
              <CheckCircle size={18} className="mr-2" />
              Start Ride
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
