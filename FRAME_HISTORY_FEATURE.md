# 📚 Frame History & Context Awareness Feature

## Overview

The CUA PDF Reader now supports **frame history** and **context awareness**, allowing the system to remember and reference content from previous frames as you scroll through a PDF.

## 🎯 Key Features

### 1. **Frame History Storage**
- Stores up to **50 frames** with their extracted content
- Each frame includes:
  - Timestamp
  - Frame image (numpy array)
  - Extracted content (text, tables, figures, highlights)
  - Frame index

### 2. **Automatic Frame Processing**
- **Auto-process toggle** in WebRTC tab
- When enabled, frames are automatically processed as you capture them
- Processing happens in background threads (non-blocking)
- Builds frame history automatically as you scroll

### 3. **Context-Aware Question Answering**
- When you ask a question, the system:
  1. Searches through frame history for relevant content
  2. Combines current frame content with historical context
  3. Provides answers using information from multiple frames

### 4. **Smart Content Search**
- Searches frame history by keywords
- Ranks results by relevance score
- Returns top 3 most relevant frames for context

## 📖 How to Use

### Building Frame History

1. **Manual Method:**
   - Go to "WebRTC Capture" tab
   - Start screen sharing
   - Scroll through your PDF
   - Periodically click "Capture Latest Frame"
   - Click "Process Captured Frame" for each frame
   - Each processed frame is added to history

2. **Automatic Method:**
   - Go to "WebRTC Capture" tab
   - Enable "🔄 Auto-process frames" checkbox
   - Click "Capture Latest Frame" periodically as you scroll
   - Frames are automatically processed in the background
   - Frame history builds automatically

### Asking Context-Aware Questions

1. Process multiple frames (manually or automatically)
2. Go to "💬 Ask Questions" tab
3. Ask questions like:
   - "What did the abstract say?" (searches all frames)
   - "Explain the method from 5 frames ago"
   - "What was the main contribution mentioned earlier?"
   - "Compare the results from the beginning of the paper"

4. The system will:
   - Search frame history for relevant content
   - Combine information from multiple frames
   - Provide comprehensive answers

## 🔍 Technical Details

### Frame History Manager (`utils/frame_history.py`)

```python
class FrameHistory:
    - add_frame(frame, extracted_content, timestamp)
    - get_frame(index)  # Get frame by index
    - get_latest_frame()  # Get most recent
    - get_recent_frames(count)  # Get last N frames
    - search_content(query, max_results)  # Search by keywords
    - get_all_text(max_chars)  # Get combined text from all frames
```

### Integration Points

1. **UI (`ui/gradio_app.py`):**
   - `CUAApp.frame_history` - FrameHistory instance
   - `auto_process_frames` - Toggle for auto-processing
   - Frame history status shown in UI

2. **Agent (`agent/cua_agent.py`):**
   - `process_query()` accepts `frame_history` parameter
   - Combines current and historical context
   - Provides comprehensive answers

3. **VLM (`utils/vlm_processor.py`):**
   - `process_frame()` extracts content from frames
   - Content stored in frame history

## 📊 Example Workflow

```
1. User opens PDF in browser
2. Starts WebRTC screen sharing
3. Enables "Auto-process frames"
4. Scrolls through PDF, clicking "Capture Latest Frame" every few seconds
   → Frame 1: Abstract (auto-processed, added to history)
   → Frame 2: Introduction (auto-processed, added to history)
   → Frame 3: Methods (auto-processed, added to history)
   → Frame 4: Results (auto-processed, added to history)
   → Frame 5: Conclusion (auto-processed, added to history)

5. User asks: "What was the main contribution mentioned in the introduction?"
   → System searches frame history
   → Finds Frame 2 (Introduction)
   → Combines with current frame context
   → Provides answer using information from Frame 2
```

## 🎨 UI Updates

### WebRTC Tab
- Added "🔄 Auto-process frames" checkbox
- Shows frame history count in status
- Displays history info after processing

### Ask Questions Tab
- Shows frame history status
- Updates after each question
- Displays how many frames are available

## 💡 Benefits

1. **No Lost Context:** Content from previous frames is remembered
2. **Better Answers:** Questions can reference earlier parts of the paper
3. **Seamless Experience:** Auto-processing builds history automatically
4. **Efficient:** Only stores last 50 frames (configurable)
5. **Smart Search:** Finds relevant frames by keyword matching

## 🔧 Configuration

- **Max Frames:** Default 50, can be changed in `FrameHistory(max_frames=50)`
- **Search Results:** Default 3 most relevant frames, configurable in `search_content()`
- **Text Limit:** Combined text limited to 10,000 chars (configurable)

## 🚀 Future Enhancements

Potential improvements:
- Automatic periodic frame capture (every N seconds)
- Semantic search instead of keyword matching
- Frame deduplication (skip similar frames)
- Export frame history to file
- Visual timeline of captured frames

