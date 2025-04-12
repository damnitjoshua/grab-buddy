"use client"

import { DriverStatus } from "@/lib/types"
import { Car, MapPin, Shield, AlertTriangle } from "lucide-react"

interface DriverPanelProps {
  driverStatus: DriverStatus
  onToggleStatus: () => void
}

export default function DriverPanel({ driverStatus, onToggleStatus }: DriverPanelProps) {
  return (
    <div className="bg-white rounded-t-2xl shadow-lg p-4 pb-8">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center">
          <div className="w-12 h-12 rounded-full bg-gray-200 mr-3 overflow-hidden">
            <img src="/placeholder.svg?height=48&width=48" alt="Driver" className="w-full h-full object-cover" />
          </div>
          <div>
            <h3 className="font-semibold">John Driver</h3>
            <div className="flex items-center">
              <span className="text-yellow-500 mr-1">★</span>
              <span className="text-sm">4.92</span>
            </div>
          </div>
        </div>
        <button
          onClick={onToggleStatus}
          className={`flex items-center justify-center rounded-full w-12 h-12 ${
            driverStatus === DriverStatus.ONLINE ? "bg-green-500 text-white" : "bg-black text-white"
          }`}
        >
          <span className="sr-only">{driverStatus === DriverStatus.ONLINE ? "Go Offline" : "Go Online"}</span>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
            className="w-6 h-6"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M5.636 5.636a9 9 0 1012.728 0M12 3v9" />
          </svg>
        </button>
      </div>

      <div className="mb-4">
        <div
          className={`flex items-center p-2 rounded-lg ${
            driverStatus === DriverStatus.ONLINE ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
          }`}
        >
          <div
            className={`w-3 h-3 rounded-full mr-2 ${
              driverStatus === DriverStatus.ONLINE ? "bg-green-500" : "bg-red-500"
            }`}
          />
          <span>{driverStatus === DriverStatus.ONLINE ? "You're online." : "You're offline."}</span>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2 mb-4">
        <div className="flex flex-col items-center justify-center bg-gray-100 rounded-lg p-3">
          <Car size={20} className="mb-1" />
          <span className="text-xs text-center">Service Types</span>
        </div>
        <div className="flex flex-col items-center justify-center bg-gray-100 rounded-lg p-3">
          <MapPin size={20} className="mb-1" />
          <span className="text-xs text-center">My Destination</span>
        </div>
        <div className="flex flex-col items-center justify-center bg-gray-100 rounded-lg p-3">
          <Shield size={20} className="mb-1" />
          <span className="text-xs text-center">Safety Center</span>
        </div>
        <div className="flex flex-col items-center justify-center bg-gray-100 rounded-lg p-3">
          <AlertTriangle size={20} className="mb-1" />
          <span className="text-xs text-center">Locations to avoid</span>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <h3 className="font-semibold">Reminder</h3>
        <svg
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={1.5}
          stroke="currentColor"
          className="w-5 h-5"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
        </svg>
      </div>
    </div>
  )
}
