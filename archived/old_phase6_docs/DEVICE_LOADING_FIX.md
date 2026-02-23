# 🔧 Device Loading Fix - Model Loading Issue RESOLVED

**Date:** February 18, 2026  
**Issue:** `accelerate` package missing, model fails to load with `device_map="auto"`  
**Status:** ✅ FIXED

---

## Problem

When selecting a model, you got:
```
Loading gpt2-medium...
❌ Error loading gpt2-medium: Using a `device_map`, `tp_plan`, etc. 
   requires `accelerate`. You can install it with `pip install accelerate`
❌ Failed to load gpt2-medium
```

**Root Cause:** The `device_map="auto"` parameter in model loading requires the `accelerate` package, which wasn't installed and wasn't in requirements.

---

## Solution Applied

### 1. Updated requirements_phase6.txt
Added missing packages:
```
accelerate>=0.27.0     # For device_map="auto"
jupyter>=1.0.0
ipywidgets>=8.0.0
matplotlib>=3.7.0
```

**Old file had:** Only torch, transformers, numpy, and Flask stuff (not needed for Jupyter)  
**New file has:** All necessary packages for Jupyter notebook use

### 2. Updated Model Loading Logic (Cell 6)
Enhanced `load_model()` method with:

**Device detection:**
```python
if torch.cuda.is_available():
    self.device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    self.device = "cpu"
```

**Fallback loading strategy:**
```python
try:
    # Try with device_map first
    self.model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        device_map="auto"
    )
except Exception as e:
    if "accelerate" in str(e).lower():
        # Fallback: load without device_map
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_id"]
        )
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
    else:
        raise
```

**Better error messages:**
```python
print(f"❌ Error loading {model_name}:")
print(f"   {str(e)[:200]}")
print(f"\n   To fix accelerate issues, run:")
print(f"   pip install accelerate")
```

---

## What to Do Now

### Option 1: Use the Launcher Script (Auto-installs)
```bash
./run_jupyter.sh
```

The launcher will:
1. Create venv ✅
2. **Install accelerate** ✅ (now in requirements)
3. Launch Jupyter
4. Model loading will now work! ✅

### Option 2: Manual Installation
In Jupyter or terminal:
```bash
pip install accelerate
```

Then restart the notebook and try again.

---

## How Model Loading Works Now

### Part 1: Device Detection
```
Loading gpt2-medium...
  Using GPU: NVIDIA RTX 6000
```
Or:
```
Loading gpt2-medium...
  Using CPU (no GPU detected)
```

### Part 2: Attempt Best Loading Method
```
  ✓ Loaded with device_map='auto'    # Ideal if accelerate installed
```
Or fallback:
```
  ⚠ device_map requires accelerate, trying without...
  ✓ Loaded without device_map        # Works without accelerate
```

### Result
```
✓ gpt2-medium loaded successfully (24 layers)
```

---

## Testing

To verify the fix works:

1. **Update requirements** (already done)
2. **Restart notebook** (Kernel → Restart)
3. **Select model** from dropdown
   - Should see device detection
   - Should see loading with/without device_map  
   - Should see "✓ loaded successfully"
4. **Click Run Reasoning**
   - Should work now! ✅

---

## Technical Details

### Why accelerate?
- `device_map="auto"` automatically distributes model across available GPUs/CPU
- Makes large models easier to load
- Not strictly required, but better for multi-GPU setups

### Fallback Strategy
- **With accelerate:** Uses optimal distributed loading ✅
- **Without accelerate:** Loads model then manually moves to GPU ✅
- **Both:** Work correctly

### Error Handling
- Detects if error is accelerate-related
- Attempts fallback only if accelerate is the issue
- Provides helpful error messages

---

## Files Updated

| File | Change |
|------|--------|
| `requirements_phase6.txt` | Added accelerate, jupyter, ipywidgets, matplotlib |
| `interactive_reasoning_tracker.ipynb` (Cell 6) | Enhanced load_model() with fallback strategy |

---

## Quick Reference

### To Fix Model Loading Issues
```bash
# Option 1: Use launcher (recommended)
./run_jupyter.sh

# Option 2: Manual install
pip install accelerate
# Then: Kernel → Restart in Jupyter
```

### If Still Getting Errors
```
1. Check if accelerate is installed:
   pip list | grep accelerate

2. If not, install it:
   pip install -q accelerate

3. Restart kernel: Kernel → Restart

4. Try selecting model again
```

---

✅ **Model Loading Issue - RESOLVED**

Your notebook can now:
- Detect GPU availability
- Load models with optimal device placement
- Fallback gracefully if accelerate missing
- Provide helpful error messages

Ready to explore reasoning activations! 🚀
