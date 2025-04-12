import type { Ride, Location } from "./types"

// Mock pickup locations
const pickupLocations: Location[] = [
  { lat: 3.1423, lng: 101.6853, address: "KLCC, Kuala Lumpur" },
  { lat: 3.1579, lng: 101.712, address: "Ampang Park, Kuala Lumpur" },
  { lat: 3.1209, lng: 101.6538, address: "Mid Valley Megamall, KL" },
  { lat: 3.1569, lng: 101.7018, address: "Great Eastern Mall, Ampang" },
]

// Mock dropoff locations
const dropoffLocations: Location[] = [
  { lat: 3.1071, lng: 101.6389, address: "KL Sentral, Kuala Lumpur" },
  { lat: 3.1349, lng: 101.6299, address: "Bangsar, Kuala Lumpur" },
  { lat: 3.0698, lng: 101.6759, address: "Sunway Pyramid, Subang Jaya" },
  { lat: 3.2039, lng: 101.6216, address: "One Utama, Petaling Jaya" },
]

// Mock passenger names
const passengerNames = ["Sarah Lee", "Ahmad Ismail", "Raj Patel", "Li Wei", "Maria Gonzalez"]

// Generate a random mock ride
export function generateMockRide(): Ride {
  const pickupIndex = Math.floor(Math.random() * pickupLocations.length)
  const dropoffIndex = Math.floor(Math.random() * dropoffLocations.length)
  const passengerIndex = Math.floor(Math.random() * passengerNames.length)

  // Calculate a random distance between 3 and 15 km
  const distance = Math.round((3 + Math.random() * 12) * 10) / 10

  // Calculate fare based on distance (base fare + per km rate)
  const fare = Math.round((5 + distance * 1.5) * 100) / 100

  // Estimate time based on distance (assuming average speed of 30 km/h)
  const estimatedTime = Math.round(distance * 2)

  return {
    id: `ride-${Date.now()}`,
    passengerName: passengerNames[passengerIndex],
    passengerRating: Math.round((3.5 + Math.random() * 1.5) * 10) / 10,
    pickupLocation: pickupLocations[pickupIndex],
    dropoffLocation: dropoffLocations[dropoffIndex],
    distance,
    fare,
    estimatedTime,
  }
}
