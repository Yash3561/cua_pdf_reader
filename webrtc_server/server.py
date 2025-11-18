"""WebRTC server for capturing screen sharing stream."""
import asyncio
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRecorder, MediaRelay
from av import VideoFrame
import queue
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameCapture:
    """Captures and stores frames from WebRTC stream."""
    
    def __init__(self, max_queue_size: int = 30):
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.latest_frame = None
        self.is_capturing = False
    
    def add_frame(self, frame: np.ndarray):
        """Add a frame to the queue."""
        self.latest_frame = frame
        
        # Add to queue, remove oldest if full
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    
    def get_latest_frame(self) -> np.ndarray:
        """Get the most recent frame."""
        return self.latest_frame
    
    def get_frame(self, timeout: float = 1.0) -> np.ndarray:
        """Get next frame from queue."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class ScreenCaptureTrack(VideoStreamTrack):
    """Custom video track that processes incoming frames."""
    
    def __init__(self, track, frame_capture: FrameCapture):
        super().__init__()
        self.track = track
        self.frame_capture = frame_capture
    
    async def recv(self):
        """Receive and process video frames."""
        frame = await self.track.recv()
        
        # Convert to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Store frame
        self.frame_capture.add_frame(img)
        
        return frame


class WebRTCServer:
    """WebRTC server for screen capture."""
    
    def __init__(self):
        self.pcs = set()
        self.frame_capture = FrameCapture()
        self.relay = MediaRelay()
    
    async def offer(self, sdp: str, type: str):
        """Handle WebRTC offer."""
        pc = RTCPeerConnection()
        self.pcs.add(pc)
        
        @pc.on("track")
        async def on_track(track):
            logger.info(f"Track received: {track.kind}")
            
            if track.kind == "video":
                # Create custom track to capture frames
                local_track = ScreenCaptureTrack(
                    self.relay.subscribe(track),
                    self.frame_capture
                )
                self.frame_capture.is_capturing = True
                logger.info("Video capture started")
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Connection state: {pc.connectionState}")
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)
        
        # Handle offer
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type=type)
        )
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }
    
    async def close(self):
        """Close all peer connections."""
        close_tasks = [pc.close() for pc in self.pcs]
        await asyncio.gather(*close_tasks)
        self.pcs.clear()
    
    def get_latest_frame(self) -> np.ndarray:
        """Get the latest captured frame."""
        return self.frame_capture.get_latest_frame()
    
    def is_capturing(self) -> bool:
        """Check if currently capturing frames."""
        return self.frame_capture.is_capturing


# Singleton instance
webrtc_server = WebRTCServer()


async def start_server():
    """Start the WebRTC server."""
    logger.info("WebRTC server ready")
    # Keep server running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        await webrtc_server.close()


if __name__ == "__main__":
    asyncio.run(start_server())