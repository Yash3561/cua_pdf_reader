import sys
print("=== Python Environment ===")
print(f"Python: {sys.version}")
print(f"Python Path: {sys.executable}\n")

# Test PyTorch
try:
    import torch
    import torchvision
    print("=== PyTorch Info ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"TorchVision: {torchvision.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ WARNING: CUDA not available - will use CPU (very slow!)")
    print("✅ PyTorch is working!\n")
except ImportError as e:
    print(f"❌ PyTorch not found: {e}\n")

# Test Semantic Scholar (no key needed)
try:
    import requests
    response = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/649def34f8be52c8b66281af98ae884c09aef38b",
        timeout=10
    )
    if response.status_code == 200:
        data = response.json()
        print("=== Semantic Scholar API ===")
        print(f"Status: ✅ Working (No API key needed!)")
        print(f"Test paper: {data.get('title', 'N/A')}")
    else:
        print(f"⚠️ API returned status: {response.status_code}")
except Exception as e:
    print(f"❌ Semantic Scholar API error: {e}")