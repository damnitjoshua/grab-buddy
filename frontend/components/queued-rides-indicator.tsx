"use client"

import type { QueuedRide } from "@/lib/types"
import { Bell } from "lucide-react"

interface QueuedRidesIndicatorProps {
  queuedRides: QueuedRide[]
  onViewRide: (index: number) => void
  showPopup: boolean
}

export default function QueuedRidesIndicator({ queuedRides, onViewRide, showPopup }: QueuedRidesIndicatorProps) {
  if (queuedRides.length === 0 && !showPopup) return null

  return (
    <div className="absolute top-16 right-4 z-20">
      {showPopup && (
        <div className="bg-green-500 text-white p-3 rounded-lg mb-2 shadow-lg animate-bounce">
          <p className="text-sm font-medium">New ride request received!</p>
        </div>
      )}

      {queuedRides.length > 0 && (
        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          <div className="bg-green-500 text-white p-2 flex items-center justify-between">
            <div className="flex items-center">
              <Bell size={16} className="mr-1" />
              <span className="text-sm font-medium">Queued Rides</span>
            </div>
            <span className="bg-white text-green-500 w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold">
              {queuedRides.length}
            </span>
          </div>

          <div className="max-h-60 overflow-y-auto">
            {queuedRides.map((ride, index) => (
              <div
                key={ride.id}
                className="p-3 border-b border-gray-100 hover:bg-gray-50 cursor-pointer"
                onClick={() => onViewRide(index)}
              >
                <div className="flex justify-between items-center">
                  <span className="font-medium">{ride.passengerName}</span>
                  <span className="text-green-500 font-medium">${ride.fare.toFixed(2)}</span>
                </div>
                <p className="text-xs text-gray-500 truncate">{ride.pickupLocation.address}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
