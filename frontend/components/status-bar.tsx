import { DriverStatus } from "@/lib/types"

interface StatusBarProps {
  driverStatus: DriverStatus
}

export default function StatusBar({ driverStatus }: StatusBarProps) {
  return (
    <div className="absolute top-0 left-0 right-0 z-10 bg-white shadow-md">
      <div className="flex items-center justify-between px-4 py-2">
        <h1 className="text-lg font-bold">GrabDriver</h1>
        <div className="flex items-center">
          <div
            className={`w-3 h-3 rounded-full mr-2 ${
              driverStatus === DriverStatus.ONLINE ? "bg-green-500" : "bg-red-500"
            }`}
          />
          <span className="text-sm font-medium">{driverStatus === DriverStatus.ONLINE ? "Online" : "Offline"}</span>
        </div>
      </div>
    </div>
  )
}
