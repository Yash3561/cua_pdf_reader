"""Complete setup test for CUA PDF Reader."""
import torch
from utils.mongodb_handler import MongoDBHandler
from utils.vlm_processor import VLMProcessor
import ollama

print("="*50)
print("COMPLETE SETUP TEST")
print("="*50)

# Test 1: GPU
print("\n1. GPU Test:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Test 2: MongoDB
print("\n2. MongoDB Test:")
try:
    db = MongoDBHandler()
    print("   ✅ MongoDB Connected")
    db.close()
except Exception as e:
    print(f"   ❌ MongoDB Error: {e}")

# Test 3: VLM
print("\n3. VLM Test:")
try:
    vlm = VLMProcessor()
    print("   ✅ VLM Loaded")
    vlm.cleanup()
except Exception as e:
    print(f"   ❌ VLM Error: {e}")

# Test 4: Ollama
print("\n4. Ollama Test:")
try:
    response = ollama.chat(
        model='qwen2.5:3b',
        messages=[{'role': 'user', 'content': 'Hi'}]
    )
    print(f"   ✅ Ollama Response: {response['message']['content'][:50]}...")
except Exception as e:
    print(f"   ❌ Ollama Error: {e}")

print("\n" + "="*50)
print("Setup test complete!")
print("="*50)