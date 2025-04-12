"use client"

import type { Ride } from "@/lib/types"
import { CheckCircle, Navigation, DollarSign, Clock } from "lucide-react"

interface RideSummaryCardProps {
  ride: Ride | null
  onDone: () => void
}

export default function RideSummaryCard({ ride, onDone }: RideSummaryCardProps) {
  if (!ride) return null

  return (
    <div className="bg-white rounded-t-2xl shadow-lg p-4">
      <div className="flex flex-col items-center justify-center mb-6">
        <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center mb-3">
          <CheckCircle size={32} className="text-green-500" />
        </div>
        <h2 className="text-xl font-bold">Ride Completed</h2>
        <p className="text-gray-500">Thanks for driving with Grab!</p>
      </div>

      <div className="border rounded-lg p-4 mb-6">
        <h3 className="font-semibold mb-3">Ride Summary</h3>

        <div className="grid grid-cols-3 gap-4 mb-4">
          <div className="flex flex-col items-center">
            <Navigation size={20} className="mb-1 text-gray-700" />
            <span className="text-sm font-medium">{ride.distance} km</span>
            <span className="text-xs text-gray-500">Distance</span>
          </div>
          <div className="flex flex-col items-center">
            <DollarSign size={20} className="mb-1 text-gray-700" />
            <span className="text-sm font-medium">${ride.fare.toFixed(2)}</span>
            <span className="text-xs text-gray-500">Fare</span>
          </div>
          <div className="flex flex-col items-center">
            <Clock size={20} className="mb-1 text-gray-700" />
            <span className="text-sm font-medium">{ride.estimatedTime} min</span>
            <span className="text-xs text-gray-500">Duration</span>
          </div>
        </div>

        <div className="border-t pt-3">
          <div className="flex justify-between mb-1">
            <span className="text-gray-600">Base fare</span>
            <span>${(ride.fare * 0.7).toFixed(2)}</span>
          </div>
          <div className="flex justify-between mb-1">
            <span className="text-gray-600">Distance fee</span>
            <span>${(ride.fare * 0.2).toFixed(2)}</span>
          </div>
          <div className="flex justify-between mb-1">
            <span className="text-gray-600">Time fee</span>
            <span>${(ride.fare * 0.1).toFixed(2)}</span>
          </div>
          <div className="flex justify-between font-semibold mt-2 pt-2 border-t">
            <span>Total earnings</span>
            <span>${ride.fare.toFixed(2)}</span>
          </div>
        </div>
      </div>

      <button onClick={onDone} className="w-full py-3 rounded-lg bg-green-500 text-white font-medium">
        Done
      </button>
    </div>
  )
}
