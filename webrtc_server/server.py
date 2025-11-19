"""WebRTC server for capturing screen sharing stream."""
import asyncio
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
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
        self.frame_lock = asyncio.Lock()
    
    async def add_frame(self, frame: np.ndarray):
        """Add a frame to the queue (thread-safe)."""
        async with self.frame_lock:
            self.latest_frame = frame.copy()  # Copy to prevent issues with reused buffers
        
        # Add to queue, remove oldest if full
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            logger.debug("Frame queue full, dropping oldest frame")
    
    async def get_latest_frame(self) -> np.ndarray:
        """Get the most recent frame."""
        async with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def get_frame_sync(self, timeout: float = 1.0) -> np.ndarray:
        """Get next frame from queue (synchronous)."""
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return frame.copy() if frame is not None else None
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
        try:
            frame = await self.track.recv()
            
            # Convert to numpy array (BGR24 for OpenCV compatibility)
            img = frame.to_ndarray(format="bgr24")
            
            # Store frame
            await self.frame_capture.add_frame(img)
            
            # Only log occasionally to reduce noise
            if not hasattr(self, '_frame_count'):
                self._frame_count = 0
            self._frame_count += 1
            if self._frame_count % 30 == 0:  # Log every 30 frames
                logger.debug(f"Frame captured: shape={img.shape}, dtype={img.dtype}")
            
            return frame
        except Exception as e:
            logger.error(f"❌ Error receiving frame: {e}", exc_info=True)
            raise


class WebRTCServer:
    """WebRTC server for screen capture."""
    
    def __init__(self):
        self.pcs = set()
        self.frame_capture = FrameCapture()
        self.relay = MediaRelay()
    
    async def offer(self, sdp: str, type: str):
        """Handle WebRTC offer."""
        try:
            pc = RTCPeerConnection()
            self.pcs.add(pc)
            
            @pc.on("track")
            async def on_track(track):
                logger.info(f"Track received: {track.kind}")
                
                if track.kind == "video":
                    logger.info(f"📹 Video track received: {track}")
                    # Create custom track to capture frames
                    local_track = ScreenCaptureTrack(
                        self.relay.subscribe(track),
                        self.frame_capture
                    )
                    self.frame_capture.is_capturing = True
                    logger.info("✅ Video capture started - waiting for frames...")
                    
                    # Start consuming frames in background
                    asyncio.create_task(self._consume_track(local_track))
            
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
            
            logger.info("WebRTC connection established, ready to receive frames")
            
            return {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            }
        except Exception as e:
            logger.error(f"Error in offer handling: {e}", exc_info=True)
            raise
    
    async def close(self):
        """Close all peer connections."""
        close_tasks = [pc.close() for pc in self.pcs]
        await asyncio.gather(*close_tasks)
        self.pcs.clear()
    
    async def get_latest_frame(self) -> np.ndarray:
        """Get the latest captured frame (async)."""
        return await self.frame_capture.get_latest_frame()
    
    def get_latest_frame_sync(self) -> np.ndarray:
        """Get the latest captured frame (synchronous)."""
        frame = self.frame_capture.latest_frame
        if frame is not None:
            logger.debug(f"📤 Returning frame: shape={frame.shape if hasattr(frame, 'shape') else 'unknown'}")
        else:
            logger.debug("⚠️ No frame available in get_latest_frame_sync")
        return frame
    
    def is_capturing(self) -> bool:
        """Check if currently capturing frames."""
        return self.frame_capture.is_capturing
    
    async def _consume_track(self, track: ScreenCaptureTrack):
        """Consume frames from track in background."""
        try:
            logger.info("🔄 Starting frame consumption loop...")
            frame_count = 0
            while True:
                try:
                    frame = await track.recv()
                    frame_count += 1
                    if frame_count % 30 == 0:  # Log every 30 frames
                        logger.info(f"📊 Frames received: {frame_count}")
                except Exception as e:
                    logger.error(f"❌ Error in frame consumption: {e}")
                    break
        except Exception as e:
            logger.error(f"❌ Frame consumption loop ended: {e}")


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