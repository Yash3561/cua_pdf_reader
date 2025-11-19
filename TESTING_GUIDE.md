# 🧪 Testing Guide - Frame Stitching & Context Awareness

## 📋 Prerequisites

1. **Start Required Services:**
   ```bash
   # Terminal 1: MongoDB (if not running as service)
   mongod
   
   # Terminal 2: Ollama (if not running as service)
   ollama serve
   ollama pull qwen2.5:3b
   
   # Terminal 3: Main Application
   python main.py
   ```

2. **Open Browser:**
   - Main UI: http://localhost:7860
   - WebRTC Client: http://localhost:8080/client.html

## 🎯 Test 1: Basic Frame Capture & Processing

### Steps:
1. **Open WebRTC Client:**
   - Go to http://localhost:8080/client.html
   - Click "Start Screen Share"
   - Select your PDF window/tab or entire screen

2. **Capture First Frame:**
   - In Gradio UI, go to "📹 WebRTC Capture" tab
   - Click "Capture Latest Frame"
   - Should see: `✅ Frame captured! Size: (1920, 1020) | History: 0 frames`
   - Click "Process Captured Frame"
   - Wait for processing (may take 10-30 seconds)
   - Should see extracted text, tables, figures

3. **Verify Frame History:**
   - Check output: `📚 Frame History: 1 frames stored`
   - Go to "💬 Ask Questions" tab
   - Check status: `📚 Frame History: 1 frames stored`

## 🎯 Test 2: Multiple Frame Processing

### Steps:
1. **Scroll PDF:**
   - In your PDF viewer, scroll down to show different content
   - Go back to Gradio UI

2. **Capture Second Frame:**
   - Click "Capture Latest Frame" again
   - Should see: `✅ Frame captured! Size: (1920, 1020) | History: 1 frames`
   - Click "Process Captured Frame"
   - Wait for processing
   - Should see: `📚 Frame History: 2 frames stored`

3. **Capture More Frames:**
   - Repeat steps 1-2 for 3-5 more frames
   - Scroll through different sections of PDF:
     - Frame 1: Title/Abstract
     - Frame 2: Introduction
     - Frame 3: Methods
     - Frame 4: Results
     - Frame 5: Conclusion
   - After each frame, verify history count increases

## 🎯 Test 3: Auto-Processing (Optional)

### Steps:
1. **Enable Auto-Processing:**
   - In "📹 WebRTC Capture" tab
   - Check "🔄 Auto-process frames" checkbox
   - Should see: `✅ Auto-processing ENABLED`

2. **Capture Frames:**
   - Scroll PDF to new section
   - Click "Capture Latest Frame"
   - Frame should be automatically processed in background
   - Check terminal for: `✅ Frame auto-processed and added to history`
   - History count should update automatically

## 🎯 Test 4: Frame Stitching

### Steps:
1. **Process Multiple Frames:**
   - Make sure you have at least 2-3 processed frames
   - Each frame should show different content from PDF

2. **Stitch Frames:**
   - Go to "💬 Ask Questions" tab
   - Find "🔗 Stitch All Frames → Reconstruct Full Document" button
   - Click the button
   - Should see output like:
     ```
     🔗 Document Stitching Complete!
     
     Method: stitched
     Frames Used: 3 / 3 (unique: 3)
     
     Stitched Document Preview:
     ---
     [First 2000 chars of combined text]
     ---
     
     💡 The full stitched document is now available for questions!
     ```

3. **Verify Stitching:**
   - Check that text from different frames is combined
   - Check that duplicates are removed
   - Check that frames are in correct order

## 🎯 Test 5: Context-Aware Questions

### Steps:
1. **Ask About Current Frame:**
   - Go to "💬 Ask Questions" tab
   - Type: "What is this paper about?"
   - Click "Ask"
   - Should get answer based on current frame content

2. **Ask About Previous Frame:**
   - Type: "What did the abstract say?"
   - Click "Ask"
   - System should search frame history
   - Should reference content from earlier frames
   - Answer should include: `[From earlier frame, relevance: X.X]`

3. **Ask About Specific Section:**
   - Type: "Explain the methods section"
   - System should find relevant frame with methods
   - Should provide comprehensive answer

4. **Ask About Full Document (After Stitching):**
   - First, stitch all frames (Test 4)
   - Then ask: "Summarize the main contributions of this paper"
   - System should use full stitched document
   - Answer should reference multiple sections

## 🎯 Test 6: Frame History Search

### Steps:
1. **Process Frames with Different Content:**
   - Frame 1: Abstract (should contain "abstract", "paper", "study")
   - Frame 2: Methods (should contain "method", "algorithm", "approach")
   - Frame 3: Results (should contain "results", "experiment", "performance")

2. **Test Search:**
   - Ask: "What methods were used?"
   - System should find Frame 2 (Methods)
   - Should show relevance score
   - Answer should reference methods section

3. **Test Multiple Matches:**
   - Ask: "What experiments were performed?"
   - System should find both Methods and Results frames
   - Should combine information from both

## 🎯 Test 7: Duplicate Detection

### Steps:
1. **Capture Same Frame Twice:**
   - Capture a frame
   - Process it
   - Without scrolling, capture same frame again
   - Process it again

2. **Verify Duplicate Handling:**
   - Stitch frames
   - Check that duplicate frame is detected
   - Check that duplicate text is removed
   - Should see: `unique_frames: 1` (if 2 identical frames)

## 🎯 Test 8: Full Workflow

### Complete End-to-End Test:

1. **Setup:**
   - Start all services
   - Open PDF in browser
   - Start WebRTC screen sharing

2. **Process PDF:**
   - Enable "Auto-process frames"
   - Scroll through PDF, capturing frames every 2-3 seconds
   - Process 5-10 frames covering different sections
   - Verify all frames are stored

3. **Stitch Document:**
   - Click "Stitch All Frames"
   - Verify stitching completes successfully
   - Check that all frames are used
   - Verify text is continuous

4. **Ask Questions:**
   - "What is the main contribution?"
   - "Explain the methodology"
   - "What were the key results?"
   - "Summarize the conclusion"
   - All should use full stitched document

5. **Verify Context:**
   - Ask about specific details from early frames
   - System should find and reference correct frames
   - Answers should be comprehensive

## 🐛 Troubleshooting

### Issue: "No frames to stitch"
- **Solution:** Process at least 2 frames first

### Issue: "Stitching failed"
- **Check:** Are frames actually different?
- **Check:** Terminal for error messages
- **Solution:** Try with more distinct frames

### Issue: "Ollama connection error"
- **Solution:** Ensure Ollama is running: `ollama serve`
- **Solution:** Check model is installed: `ollama pull qwen2.5:3b`

### Issue: "WebRTC not capturing"
- **Solution:** Restart screen sharing
- **Solution:** Try sharing entire screen instead of tab
- **Solution:** Check browser console for errors

### Issue: "Frames not being stored"
- **Check:** Terminal for processing errors
- **Solution:** Manually process frames instead of auto-processing
- **Solution:** Check frame history count in UI

## ✅ Success Criteria

After testing, you should be able to:

1. ✅ Capture and process multiple frames
2. ✅ See frame history count increase
3. ✅ Stitch frames into full document
4. ✅ Ask questions about previous frames
5. ✅ Get answers using full stitched document
6. ✅ See context from multiple frames in answers

## 📊 Expected Output Examples

### After Processing 3 Frames:
```
📚 Frame History: 3 frames stored
💡 Context Awareness: You can ask questions about content from previous frames!
🔗 Document Stitching: 3 frames stitched together
   Full document reconstruction available!
```

### After Stitching:
```
🔗 Document Stitching Complete!

Method: stitched
Frames Used: 3 / 3 (unique: 3)

Stitched Document Preview:
---
[Combined text from all 3 frames, no duplicates]
---
```

### Question Answer:
```
User: What did the abstract say?

Assistant: Based on the abstract from an earlier frame (relevance: 8.5), 
the paper presents a comprehensive study on using deep reinforcement 
learning to create dynamic locomotion controllers for bipedal robots...
```

## 🎓 Next Steps

Once basic testing works:
1. Test with longer PDFs (10+ pages)
2. Test with different PDF layouts
3. Test scrolling speed variations
4. Test with PDFs containing many figures/tables
5. Test edge cases (very similar frames, rapid scrolling)

