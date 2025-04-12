export enum DriverStatus {
  OFFLINE = "OFFLINE",
  ONLINE = "ONLINE",
}

export enum RideStatus {
  IDLE = "IDLE",
  REQUESTED = "REQUESTED",
  ACCEPTED = "ACCEPTED",
  PICKUP_REACHED = "PICKUP_REACHED",
  IN_PROGRESS = "IN_PROGRESS",
  COMPLETED = "COMPLETED",
  CONFIRMING_NEXT_RIDE = "CONFIRMING_NEXT_RIDE", // New status for confirming next ride
}

export interface Location {
  lat: number
  lng: number
  address: string
}

export interface Ride {
  id: string
  passengerName: string
  passengerRating: number
  pickupLocation: Location
  dropoffLocation: Location
  distance: number
  fare: number
  estimatedTime: number
  timestamp?: Date // When the request was received
}

export interface Message {
  id: string
  text: string
  isDriver: boolean
  timestamp: Date
}

export interface QueuedRide extends Ride {
  status: "pending" | "accepted" | "expired"
  timestamp: Date
}
