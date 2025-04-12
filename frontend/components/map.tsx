"use client"

import { useEffect, useRef, useState } from "react"
import dynamic from "next/dynamic"
import L from "leaflet"
import "leaflet/dist/leaflet.css"
import { type DriverStatus, RideStatus, type Ride } from "@/lib/types"

// Dynamically import react-leaflet components to avoid SSR issues
const MapContainer = dynamic(() => import("react-leaflet").then((mod) => mod.MapContainer), { ssr: false })
const TileLayer = dynamic(() => import("react-leaflet").then((mod) => mod.TileLayer), { ssr: false })
const Marker = dynamic(() => import("react-leaflet").then((mod) => mod.Marker), { ssr: false })
const Popup = dynamic(() => import("react-leaflet").then((mod) => mod.Popup), { ssr: false })
const Polyline = dynamic(() => import("react-leaflet").then((mod) => mod.Polyline), { ssr: false })

// Custom component to handle map location updates
const MapUpdater = dynamic(
  () =>
    import("react-leaflet").then((mod) => {
      const { useMap } = mod
      return function MapUpdater({ center }: { center: [number, number] }) {
        const map = useMap()
        useEffect(() => {
          map.setView(center, map.getZoom())
        }, [center, map])
        return null
      }
    }),
  { ssr: false },
)

// Create custom marker icons using inline SVG to avoid broken image issues
const createIcons = () => {
  // Driver marker (green circle with plus)
  const driverIcon = new L.DivIcon({
    html: `<div style="background-color: #1f9d55; width: 24px; height: 24px; border-radius: 50%; border: 2px solid white; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">+</div>`,
    className: "custom-div-icon",
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  })

  // Pickup marker (green circle)
  const pickupIcon = new L.DivIcon({
    html: `<div style="background-color: #4ade80; width: 24px; height: 24px; border-radius: 50%; border: 2px solid white; display: flex; align-items: center; justify-content: center;"></div>`,
    className: "custom-div-icon",
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  })

  // Dropoff marker (red circle)
  const dropoffIcon = new L.DivIcon({
    html: `<div style="background-color: #ef4444; width: 24px; height: 24px; border-radius: 50%; border: 2px solid white; display: flex; align-items: center; justify-content: center;"></div>`,
    className: "custom-div-icon",
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  })

  return { driverIcon, pickupIcon, dropoffIcon }
}

interface MapProps {
  driverStatus: DriverStatus
  rideStatus: RideStatus
  currentRide: Ride | null
}

export default function Map({ driverStatus, rideStatus, currentRide }: MapProps) {
  // Default to Kuala Lumpur city center
  const [position, setPosition] = useState<[number, number]>([3.139, 101.6869])
  const [routeToPickup, setRouteToPickup] = useState<[number, number][]>([])
  const [routeToDropoff, setRouteToDropoff] = useState<[number, number][]>([])
  const mapRef = useRef<L.Map | null>(null)
  const [icons, setIcons] = useState<{ driverIcon: L.DivIcon; pickupIcon: L.DivIcon; dropoffIcon: L.DivIcon } | null>(
    null,
  )

  // Initialize Leaflet icons
  useEffect(() => {
    setIcons(createIcons())
  }, [])

  // Try to get user's location, but handle the case where it's not available
  useEffect(() => {
    // First set a random position near the default location to simulate movement
    // This ensures the app works even without geolocation
    const randomOffset = () => (Math.random() - 0.5) * 0.01
    setPosition([3.139 + randomOffset(), 101.6869 + randomOffset()])

    // Then try to get the actual location if available
    if (navigator.geolocation) {
      try {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            const { latitude, longitude } = position.coords
            setPosition([latitude, longitude])
          },
          (error) => {
            console.error("Error getting location:", error)
            // Already using default position, so no need to set it again
          },
          { timeout: 5000, enableHighAccuracy: true },
        )
      } catch (error) {
        console.error("Geolocation error:", error)
        // Already using default position, so no need to set it again
      }
    }
  }, [])

  // Generate mock routes when ride is accepted
  useEffect(() => {
    let mockRouteToPickup: [number, number][] = []
    let mockRouteToDropoff: [number, number][] = []

    if (rideStatus === RideStatus.ACCEPTED && currentRide) {
      // Generate mock route to pickup
      mockRouteToPickup = generateMockRoute(
        position,
        [currentRide.pickupLocation.lat, currentRide.pickupLocation.lng],
        5,
      )

      // Generate mock route from pickup to dropoff
      mockRouteToDropoff = generateMockRoute(
        [currentRide.pickupLocation.lat, currentRide.pickupLocation.lng],
        [currentRide.dropoffLocation.lat, currentRide.dropoffLocation.lng],
        8,
      )
    }

    setRouteToPickup(mockRouteToPickup)
    setRouteToDropoff(mockRouteToDropoff)
  }, [rideStatus, currentRide, position])

  // Helper function to generate a mock route between two points
  function generateMockRoute(start: [number, number], end: [number, number], points: number): [number, number][] {
    const route: [number, number][] = [start]

    for (let i = 1; i <= points; i++) {
      const ratio = i / (points + 1)
      const lat = start[0] + (end[0] - start[0]) * ratio + (Math.random() - 0.5) * 0.005
      const lng = start[1] + (end[1] - start[1]) * ratio + (Math.random() - 0.5) * 0.005
      route.push([lat, lng])
    }

    route.push(end)
    return route
  }

  // If icons aren't loaded yet or we're on the server, render a loading placeholder
  if (!icons) {
    return <div className="h-screen w-full bg-gray-100"></div>
  }

  return (
    <MapContainer
      center={position}
      zoom={15}
      style={{ height: "100vh", width: "100%" }}
      zoomControl={false}
      whenCreated={(map) => {
        mapRef.current = map
      }}
    >
      {/* Use standard OpenStreetMap tiles that don't require authentication */}
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      <MapUpdater center={position} />

      {/* Driver marker */}
      <Marker position={position} icon={icons.driverIcon}>
        <Popup>You are here</Popup>
      </Marker>

      {/* Pickup location marker */}
      {currentRide && rideStatus !== RideStatus.IDLE && rideStatus !== RideStatus.COMPLETED && (
        <Marker position={[currentRide.pickupLocation.lat, currentRide.pickupLocation.lng]} icon={icons.pickupIcon}>
          <Popup>Pickup: {currentRide.pickupLocation.address}</Popup>
        </Marker>
      )}

      {/* Dropoff location marker */}
      {currentRide && rideStatus !== RideStatus.IDLE && rideStatus !== RideStatus.COMPLETED && (
        <Marker position={[currentRide.dropoffLocation.lat, currentRide.dropoffLocation.lng]} icon={icons.dropoffIcon}>
          <Popup>Dropoff: {currentRide.dropoffLocation.address}</Popup>
        </Marker>
      )}

      {/* Route to pickup */}
      {rideStatus === RideStatus.ACCEPTED && (
        <Polyline positions={routeToPickup} color="#1f9d55" weight={4} opacity={0.7} dashArray="10, 10" />
      )}

      {/* Route to dropoff */}
      {(rideStatus === RideStatus.PICKUP_REACHED || rideStatus === RideStatus.IN_PROGRESS) && (
        <Polyline positions={routeToDropoff} color="#1f9d55" weight={4} opacity={0.7} />
      )}
    </MapContainer>
  )
}
