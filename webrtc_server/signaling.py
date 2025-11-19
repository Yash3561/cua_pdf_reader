"""FastAPI signaling server for WebRTC connections."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import logging
import os
from pathlib import Path
from .server import webrtc_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CUA WebRTC Signaling Server")

# Enable CORS for browser connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class OfferRequest(BaseModel):
    """WebRTC offer request."""
    sdp: str
    type: str


class AnswerResponse(BaseModel):
    """WebRTC answer response."""
    sdp: str
    type: str


@app.post("/offer", response_model=AnswerResponse)
async def handle_offer(offer: OfferRequest):
    """Handle WebRTC offer from browser and return answer."""
    try:
        logger.info(f"Received WebRTC offer: {offer.type}")
        answer = await webrtc_server.offer(offer.sdp, offer.type)
        logger.info("WebRTC answer created successfully")
        return AnswerResponse(sdp=answer["sdp"], type=answer["type"])
    except Exception as e:
        logger.error(f"Error handling offer: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_status():
    """Get WebRTC connection status."""
    return {
        "is_capturing": webrtc_server.is_capturing(),
        "has_frame": webrtc_server.get_latest_frame_sync() is not None
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/client.html")
async def get_client():
    """Serve the WebRTC client HTML page."""
    client_path = Path(__file__).parent / "client.html"
    if client_path.exists():
        return FileResponse(client_path)
    else:
        raise HTTPException(status_code=404, detail="Client HTML not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

