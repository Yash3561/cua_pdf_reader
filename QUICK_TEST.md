# ⚡ Quick Test Guide - Frame Stitching

## 🚀 Quick Start (5 minutes)

### Step 1: Start Services
```bash
# Terminal 1: Start MongoDB (if needed)
mongod

# Terminal 2: Start Ollama (if needed)  
ollama serve
ollama pull qwen2.5:3b

# Terminal 3: Start App
python main.py
```

### Step 2: Setup Screen Sharing
1. Open http://localhost:8080/client.html in a new tab
2. Click "Start Screen Share"
3. Select your PDF window/tab

### Step 3: Process Multiple Frames
1. Open http://localhost:7860 in browser
2. Go to "📹 WebRTC Capture" tab
3. **Capture Frame 1:**
   - Click "Capture Latest Frame"
   - Click "Process Captured Frame"
   - Wait ~20 seconds
   - Should see: `📚 Frame History: 1 frames stored`

4. **Scroll PDF down** to show different content

5. **Capture Frame 2:**
   - Click "Capture Latest Frame" again
   - Click "Process Captured Frame"
   - Should see: `📚 Frame History: 2 frames stored`

6. **Repeat 2-3 more times** to get 4-5 frames total

### Step 4: Test Stitching
1. Go to "💬 Ask Questions" tab
2. Click "🔗 Stitch All Frames → Reconstruct Full Document"
3. Should see:
   ```
   🔗 Document Stitching Complete!
   Method: stitched
   Frames Used: 4 / 4
   [Preview of stitched text]
   ```

### Step 5: Test Context-Aware Questions
1. In "💬 Ask Questions" tab, type: **"What did the abstract say?"**
2. Click "Ask"
3. System should find and reference earlier frames
4. Answer should include content from abstract frame

## ✅ Success Indicators

- ✅ Frame history count increases with each processed frame
- ✅ Stitching shows "Method: stitched" and uses all frames
- ✅ Questions about previous frames get answered correctly
- ✅ Stitched document preview shows combined text

## 🐛 Quick Fixes

**"Ollama connection error"** → Run `ollama serve` in separate terminal

**"No frames to stitch"** → Process at least 2 frames first

**"WebRTC not capturing"** → Restart screen sharing, try sharing entire screen

**"Stitching failed"** → Make sure frames have different content (scroll between captures)

