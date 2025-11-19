"""Simple test script to debug WebRTC frame capture."""
import asyncio
import requests
from webrtc_server.server import webrtc_server
import time

def test_status():
    """Test the status endpoint."""
    print("=" * 50)
    print("STEP 1: Testing Status Endpoint")
    print("=" * 50)
    try:
        resp = requests.get("http://localhost:8080/status", timeout=2)
        data = resp.json()
        print(f"✅ Status endpoint works!")
        print(f"   is_capturing: {data.get('is_capturing')}")
        print(f"   has_frame: {data.get('has_frame')}")
        return True
    except Exception as e:
        print(f"❌ Status endpoint failed: {e}")
        return False

def test_frame_capture():
    """Test frame capture."""
    print("\n" + "=" * 50)
    print("STEP 2: Testing Frame Capture")
    print("=" * 50)
    
    print(f"📹 is_capturing: {webrtc_server.is_capturing()}")
    
    frame = webrtc_server.get_latest_frame_sync()
    if frame is not None:
        print(f"✅ Frame captured!")
        print(f"   Type: {type(frame)}")
        print(f"   Shape: {frame.shape}")
        print(f"   Dtype: {frame.dtype}")
        return True
    else:
        print("❌ No frame available")
        print("   Make sure:")
        print("   1. Screen sharing is active in browser")
        print("   2. WebRTC connection is established")
        print("   3. Wait a few seconds for frames to arrive")
        return False

def main():
    """Run all tests."""
    print("\n🔍 WebRTC Frame Capture Debug Test\n")
    print("Instructions:")
    print("1. Make sure the server is running (python main.py)")
    print("2. Open http://localhost:8080/client.html in browser")
    print("3. Click 'Start Screen Share'")
    print("4. Select a window to share")
    print("5. Wait for connection to establish")
    print("6. Run this test script\n")
    
    input("Press Enter when screen sharing is active...")
    
    # Test 1: Status endpoint
    if not test_status():
        print("\n❌ Status test failed. Check if server is running.")
        return
    
    # Wait a bit for frames
    print("\n⏳ Waiting 3 seconds for frames to arrive...")
    time.sleep(3)
    
    # Test 2: Frame capture
    test_frame_capture()
    
    print("\n" + "=" * 50)
    print("Debug test complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()

