"use client"

import { Bell } from "lucide-react"

interface NotificationFabProps {
  count: number
  onClick: () => void
  showAnimation: boolean
}

export default function NotificationFab({ count, onClick, showAnimation }: NotificationFabProps) {
  return (
    <button
      onClick={onClick}
      className={`fixed top-16 right-4 z-30 bg-green-500 text-white rounded-full w-12 h-12 flex items-center justify-center shadow-lg ${
        showAnimation ? "animate-bounce" : ""
      }`}
      aria-label="View queued ride requests"
    >
      <Bell size={20} />
      {count > 0 && (
        <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
          {count}
        </span>
      )}
    </button>
  )
}
