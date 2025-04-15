"use client"
import dynamic from "next/dynamic" // Import dynamic
import StatusBar from "@/components/status-bar"
import DriverPanel from "@/components/driver-panel"
import RideRequestCard from "@/components/ride-request-card"
import RideInProgressCard from "@/components/ride-in-progress-card"
import RideSummaryCard from "@/components/ride-summary-card"
import QueuedRidesPanel from "@/components/queued-rides-panel"
import NotificationFab from "@/components/notification-fab"
import RideConfirmationModal from "@/components/ride-confirmation-modal"
import PendingRequestsAfterRidePanel from "@/components/pending-requests-after-ride-panel"
import VoiceCommandIndicator from "@/components/voice-command-indicator"
import { RideStatus } from "@/lib/types"
import { useDriverState } from "@/lib/driver-state"
import { useVoiceCommands } from "@/lib/use-voice-commands"

// Dynamically import the Map component with SSR disabled
const DynamicMap = dynamic(() => import("@/components/map"), {
  ssr: false,
  loading: () => <div className="h-screen w-full bg-gray-200 flex items-center justify-center">Loading Map...</div>,
})

export default function Home() {
  const {
    driverStatus,
    rideStatus,
    currentRide,
    nextRide,
    messages,
    rideRequests,
    currentRequestIndex,
    queuedRides,
    pendingQueuedRidesCount,
    showNewRequestPopup,
    showQueuedRidesPanel,
    showPendingRequestsAfterRide,
    toggleDriverStatus,
    toggleQueuedRidesPanel,
    closePendingRequestsPanel,
    acceptRide,
    declineRide,
    declineAllRides,
    acceptQueuedRide,
    declineQueuedRide,
    startRide,
    endRide,
    resetRide,
    acceptNextRide,
    declineNextRide,
    sendMessage,
    nextRequest,
    prevRequest,
    viewQueuedRide,
  } = useDriverState()

  // Initialize voice commands
  const { isSupported, isListeningForWakeWord, isVoiceCommandActive, lastCommand, error, activateVoiceCommands } =
    useVoiceCommands({
      driverStatus,
      rideStatus,
      toggleDriverStatus,
      acceptRide,
      declineRide,
      startRide,
      endRide,
      toggleQueuedRidesPanel,
    })

  return (
    <main className="relative h-screen w-full overflow-hidden">
      {/* Use the dynamically imported Map component */}
      <DynamicMap driverStatus={driverStatus} rideStatus={rideStatus} currentRide={currentRide} />

      <StatusBar driverStatus={driverStatus} />

      {/* Voice Command Indicator */}
      <VoiceCommandIndicator
        isListeningForWakeWord={isListeningForWakeWord}
        isVoiceCommandActive={isVoiceCommandActive}
        lastCommand={lastCommand}
        error={error}
        onActivate={activateVoiceCommands}
      />

      {/* Notification FAB for queued rides - only show during active rides */}
      {(rideStatus === RideStatus.ACCEPTED ||
        rideStatus === RideStatus.PICKUP_REACHED ||
        rideStatus === RideStatus.IN_PROGRESS) && (
        <NotificationFab
          count={pendingQueuedRidesCount}
          onClick={toggleQueuedRidesPanel}
          showAnimation={showNewRequestPopup}
        />
      )}

      {/* Queued Rides Panel */}
      <QueuedRidesPanel
        queuedRides={queuedRides}
        onViewRide={viewQueuedRide}
        onAccept={acceptQueuedRide}
        onDecline={declineQueuedRide}
        onClose={toggleQueuedRidesPanel}
        isOpen={showQueuedRidesPanel}
      />
      {/* Pending Requests After Ride Panel */}
      <PendingRequestsAfterRidePanel
        pendingRides={queuedRides.filter((ride) => ride.status === "pending")}
        onAccept={acceptQueuedRide}
        onDecline={declineQueuedRide}
        onClose={closePendingRequestsPanel}
        isOpen={showPendingRequestsAfterRide}
      />

      {/* Ride Confirmation Modal - shown after completing a ride when there's a previously accepted ride */}
      {rideStatus === RideStatus.CONFIRMING_NEXT_RIDE && nextRide && (
        <RideConfirmationModal ride={nextRide} onAccept={acceptNextRide} onDecline={declineNextRide} />
      )}

      <div className="absolute bottom-0 left-0 right-0 z-10">
        {rideStatus === RideStatus.REQUESTED && (
          <RideRequestCard
            rides={rideRequests}
            currentIndex={currentRequestIndex}
            onAccept={acceptRide}
            onDecline={declineRide}
            onDeclineAll={declineAllRides}
            onNext={nextRequest}
            onPrev={prevRequest}
          />
        )}

        {(rideStatus === RideStatus.ACCEPTED ||
          rideStatus === RideStatus.PICKUP_REACHED ||
          rideStatus === RideStatus.IN_PROGRESS) && (
          <RideInProgressCard
            ride={currentRide}
            rideStatus={rideStatus}
            messages={messages}
            onStartRide={startRide}
            onEndRide={endRide}
            onSendMessage={sendMessage}
          />
        )}

        {rideStatus === RideStatus.COMPLETED && <RideSummaryCard ride={currentRide} onDone={resetRide} />}

        {rideStatus === RideStatus.IDLE && (
          <DriverPanel driverStatus={driverStatus} onToggleStatus={toggleDriverStatus} />
        )}
      </div>
    </main>
  )
}
