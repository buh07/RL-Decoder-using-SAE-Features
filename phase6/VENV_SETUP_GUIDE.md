# 🔍 Jupyter Notebook Virtual Environment Guide

## How the Notebook Activates Virtual Environment

The notebook now includes automatic venv detection and setup:

### Cell 1: Kernel Diagnostics (Automatic)
```
🚀 JUPYTER KERNEL STARTUP DIAGNOSTICS
Kernel Python: /path/to/venv/bin/python
Version: 3.10.x
Platform: linux
Running in Virtual Environment: ✅ YES
```

**What it does:**
- ✅ Shows which Python kernel is running
- ✅ Detects if venv is active
- ⚠️ Warns if venv not detected + shows how to fix

### Cell 2: Virtual Environment Verification
```
🔍 VIRTUAL ENVIRONMENT SETUP
Virtual environment status: ✅ ACTIVE
Python executable: /path/to/venv/bin/python
Virtual env path: /path/to/venv
```

**What it does:**
- ✅ Confirms venv is active
- ✅ Shows exact venv location
- ⚠️ If not active: shows guidance to fix
- 📦 Imports all core libraries
- 🔧 Auto-installs if missing

### Cell 3: ML Libraries Setup
```
📥 LOADING ML LIBRARIES
✅ Transformers imported successfully
✅ SAE architecture imported successfully
Using Python: /path/to/venv/bin/python
```

**What it does:**
- ✅ Loads transformers, SAE modules
- ⚠️ Auto-installs if missing
- 🐍 Always uses kernel's Python (from venv)

---

## Two Ways to Run

### ✅ Recommended: Launcher Script (Automatic venv)

```bash
cd phase6
./run_jupyter.sh
```

**The launcher script:**
1. Creates venv (if needed)
2. Activates venv
3. Installs dependencies in venv
4. Launches Jupyter with venv activated
5. Notebook runs in venv kernel automatically

**Result:** All cells see venv automatically! ✅

---

### Manual: Activate venv, then launch

```bash
# Step 1: Create venv (one time)
python3 -m venv venv

# Step 2: Activate venv (every time)
source venv/bin/activate

# Step 3: Install dependencies (one time)
pip install -r requirements_phase6.txt

# Step 4: Launch notebook
jupyter notebook interactive_reasoning_tracker.ipynb
```

**Result:** Notebook runs in venv kernel ✅

---

## What Each Cell Does for venv

| Cell | Does | Uses venv? |
|------|------|-----------|
| 1 | Kernel diagnostics | Shows if active |
| 2 | Core libs + venv verification | Detects venv, installs via `sys.executable` |
| 3 | Transformers + SAE libs | Installs via `sys.executable` (uses venv) |
| 4+ | Model config + inference | All run in venv kernel |

---

## venv Awareness Features

### 1. Detects venv Status
```python
venv_active = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
```

### 2. Shows Exact Python Path
```python
print(f"Python executable: {sys.executable}")
```

### 3. Uses Kernel's Python for Installs
```python
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "package"])
```

This ensures:
- ✅ If kernel is venv, installs to venv
- ✅ If kernel is system, warns user
- ✅ Always installs to correct location

### 4. Provides Guidance if Not in venv

**Example output:**
```
⚠️  IMPORTANT: Virtual environment not detected!

TO FIX: Run this notebook with the launcher script:
  cd phase6
  ./run_jupyter.sh

If running manually, activate venv first:
  source venv/bin/activate
  jupyter notebook interactive_reasoning_tracker.ipynb
```

---

## Troubleshooting

### "Not in venv" warning but wants to proceed

**This is OK** - The notebook will:
1. ✅ Still work (just less isolated)
2. ✅ Install packages to kernel
3. ⚠️ If system Python, ask sudo (blocked in Jupyter)

**Solution:** Use launcher script
```bash
./run_jupyter.sh
```

---

### Kernel uses wrong Python

**Check which kernel is active:**
```python
# In notebook cell
import sys
print(sys.executable)
```

**Should show:** `/path/to/phase6/venv/bin/python`

**If shows:** `/usr/bin/python3` or similar
- You're using system Python, not venv
- Restart Jupyter with venv activated (use launcher script)

---

### Can't install because "externally managed"

✅ **No problem** - The launcher script handles this!

The launcher creates a venv, and:
- ✅ Installs to venv (not system)
- ✅ No "externally managed" error
- ✅ Clean isolated environment

---

## Manual venv Workflow

If you prefer to manage venv yourself:

```bash
# Terminal 1: Setup and launch Jupyter
cd /scratch2/f004ndc/RL-Decoder\ with\ SAE\ Features/phase6
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_phase6.txt
jupyter notebook interactive_reasoning_tracker.ipynb
# Jupyter opens on http://localhost:8888

# Terminal 2 (optional): Check venv in notebook
# Run Cell 3 and look for:
# "Python executable: /path/to/venv/bin/python"
```

---

## Remote SSH + venv

### Option 1: Use launcher (easiest)
```bash
# SSH into remote
ssh user@remote

# Launch notebook with venv
cd phase6
./run_jupyter.sh
# Shows: http://127.0.0.1:8888/?token=abc123

# On local browser: paste URL
# http://127.0.0.1:8888/?token=abc123
```

### Option 2: Port forwarding + launcher
```bash
# Local machine
ssh -L 8888:localhost:8888 user@remote

# After SSH connects, on remote:
cd phase6
./run_jupyter.sh

# On local browser: http://localhost:8888
```

---

## What Happens at Startup

### First Run (with launcher)
```
📦 Creating virtual environment...        [2-3 sec]
✓ Virtual environment created

📥 Ensuring dependencies...               [30-40 sec]
Installing Jupyter...
Installing ML dependencies...
✓ All dependencies ready

🚀 Starting Jupyter notebook...           [2-3 sec]
[Jupyter launches]

# Browser opens automatically
```

### Subsequent Runs
```
📥 Ensuring dependencies...               [1-2 sec]
✓ All dependencies ready

🚀 Starting Jupyter notebook...           [2-3 sec]
[Jupyter launches immediately]
```

---

## Summary

✅ **The notebook now:**
1. Automatically detects venv status on startup
2. Shows exact Python kernel location
3. Provides guidance if venv not found
4. Installs packages to venv (via `sys.executable`)
5. Works with both launcher script and manual setup
6. Handles "externally managed" errors gracefully
7. Works over SSH with port forwarding

✅ **To use it:**
```bash
cd phase6
./run_jupyter.sh
```

That's it! The venv is activated automatically.
