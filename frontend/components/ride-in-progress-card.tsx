"use client"

import { useState } from "react"
import { type Ride, RideStatus, type Message } from "@/lib/types"
import { Phone, MessageSquare, Send } from "lucide-react"

interface RideInProgressCardProps {
  ride: Ride | null
  rideStatus: RideStatus
  messages: Message[]
  onStartRide: () => void
  onEndRide: () => void
  onSendMessage: (text: string) => void
}

export default function RideInProgressCard({
  ride,
  rideStatus,
  messages,
  onStartRide,
  onEndRide,
  onSendMessage,
}: RideInProgressCardProps) {
  const [showChat, setShowChat] = useState(false)
  const [messageText, setMessageText] = useState("")

  if (!ride) return null

  const handleSendMessage = () => {
    if (messageText.trim()) {
      onSendMessage(messageText)
      setMessageText("")
    }
  }

  return (
    <div className="bg-white rounded-t-2xl shadow-lg p-4">
      {showChat ? (
        <div className="h-[350px] flex flex-col">
          <div className="flex justify-between items-center mb-3">
            <div className="flex items-center">
              <button onClick={() => setShowChat(false)} className="mr-3 p-1 rounded-full hover:bg-gray-100">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.5}
                  stroke="currentColor"
                  className="w-5 h-5"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 19.5L8.25 12l7.5-7.5" />
                </svg>
              </button>
              <div className="flex items-center">
                <div className="w-8 h-8 rounded-full bg-gray-200 overflow-hidden mr-2">
                  <img
                    src="/placeholder.svg?height=32&width=32"
                    alt="Passenger"
                    className="w-full h-full object-cover"
                  />
                </div>
                <span className="font-medium">{ride.passengerName}</span>
              </div>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto mb-3 space-y-2">
            {messages.length === 0 ? (
              <div className="text-center text-gray-500 py-4">No messages yet</div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`max-w-[80%] p-2 rounded-lg ${
                    message.isDriver
                      ? "bg-green-100 text-green-800 ml-auto rounded-tr-none"
                      : "bg-gray-100 text-gray-800 rounded-tl-none"
                  }`}
                >
                  <p>{message.text}</p>
                  <p className="text-xs text-gray-500 mt-1">
                    {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </p>
                </div>
              ))
            )}
          </div>

          <div className="flex items-center gap-2">
            <input
              type="text"
              value={messageText}
              onChange={(e) => setMessageText(e.target.value)}
              placeholder="Type a message..."
              className="flex-1 border border-gray-300 rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-green-500"
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  handleSendMessage()
                }
              }}
            />
            <button
              onClick={handleSendMessage}
              className="w-10 h-10 rounded-full bg-green-500 text-white flex items-center justify-center"
            >
              <Send size={18} />
            </button>
          </div>
        </div>
      ) : (
        <>
          <div className="flex justify-between items-center mb-4">
            <div>
              <h2 className="text-lg font-bold">
                {rideStatus === RideStatus.ACCEPTED && "Heading to pickup"}
                {rideStatus === RideStatus.PICKUP_REACHED && "Arrived at pickup"}
                {rideStatus === RideStatus.IN_PROGRESS && "Ride in progress"}
              </h2>
              <p className="text-sm text-gray-500">
                {rideStatus === RideStatus.ACCEPTED && `${ride.estimatedTime} min away`}
                {rideStatus === RideStatus.PICKUP_REACHED && "Passenger is waiting"}
                {rideStatus === RideStatus.IN_PROGRESS && `${ride.estimatedTime} min to destination`}
              </p>
            </div>
            <div className="flex items-center">
              <div className="w-10 h-10 rounded-full bg-gray-200 overflow-hidden">
                <img src="/placeholder.svg?height=40&width=40" alt="Passenger" className="w-full h-full object-cover" />
              </div>
              <div className="ml-2">
                <p className="font-medium">{ride.passengerName}</p>
                <div className="flex items-center">
                  <span className="text-yellow-500 mr-1">★</span>
                  <span className="text-sm">{ride.passengerRating}</span>
                </div>
              </div>
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

          <div className="flex gap-3 mb-4">
            <button className="flex-1 flex items-center justify-center py-2 rounded-lg border border-gray-300">
              <Phone size={18} className="mr-2" />
              <span>Call</span>
            </button>
            <button
              onClick={() => setShowChat(true)}
              className="flex-1 flex items-center justify-center py-2 rounded-lg border border-gray-300"
            >
              <MessageSquare size={18} className="mr-2" />
              <span>Message</span>
              {messages.length > 0 && (
                <span className="ml-1 bg-green-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                  {messages.length}
                </span>
              )}
            </button>
          </div>

          {rideStatus === RideStatus.ACCEPTED && (
            <button onClick={onStartRide} className="w-full py-3 rounded-lg bg-green-500 text-white font-medium">
              Arrived at Pickup
            </button>
          )}

          {rideStatus === RideStatus.PICKUP_REACHED && (
            <button onClick={onStartRide} className="w-full py-3 rounded-lg bg-green-500 text-white font-medium">
              Start Ride
            </button>
          )}

          {rideStatus === RideStatus.IN_PROGRESS && (
            <button onClick={onEndRide} className="w-full py-3 rounded-lg bg-green-500 text-white font-medium">
              End Ride
            </button>
          )}
        </>
      )}
    </div>
  )
}
