# 🚀 Quick Start - Jupyter Notebook with Virtual Environment

## One-Command Launch

```bash
cd phase6
./run_jupyter.sh
```

That's it! The script will:
1. ✅ Create virtual environment (if needed)
2. ✅ Activate venv
3. ✅ Install dependencies in venv
4. ✅ Launch Jupyter notebook

Browser will automatically open at `http://localhost:8888`

---

## What Happens

### First Run
```
📦 Creating virtual environment...
✓ Virtual environment created

📥 Ensuring dependencies...
Installing Jupyter...
Installing ML dependencies (this may take a minute)...
✓ All dependencies ready

🚀 Starting Jupyter notebook...
[Jupyter starts]
```

### Subsequent Runs
```
📥 Ensuring dependencies...
✓ All dependencies ready

🚀 Starting Jupyter notebook...
[Jupyter starts immediately, ~5 seconds]
```

---

## Custom Port (Optional)

```bash
./run_jupyter.sh 8889
# Opens on http://localhost:8889
```

---

## Remote SSH Usage

### Option 1: Direct URL Copy (Recommended)

**On remote machine:**
```bash
./run_jupyter.sh
# Prints:
# [I 16:45:23.123 NotebookApp] The Jupyter Notebook is running at:
# http://127.0.0.1:8888/?token=abc123xyz...
```

**On your local machine:**
- Copy the full URL from remote (including token)
- Paste into local browser

### Option 2: SSH Tunneling

**On your local machine:**
```bash
ssh -L 8888:localhost:8888 username@remote-host
```

**Then on remote (after SSH connects):**
```bash
cd /path/to/phase6
./run_jupyter.sh
```

**On local browser:**
- Open `http://localhost:8888`
- Enter token if prompted (from remote terminal output)

---

## Inside the Notebook

1. **Cell 1**: Read the introduction
2. **Cell 2**: Imports (auto-installs missing packages if needed)
3. **Cell 3**: Load Transformers/SAE architecture
4. **Cell 4**: Define model configurations
5. **Cell 5**: Initialize tracker
6. **Cells 6-10**: Interactive controls
7. **Cell 11+**: Run inference!

### First-Time Use

- Cell 1: ~1-2 sec
- Cell 2: ~10 sec (installs libs in venv)
- Cell 3: ~5 sec
- Cell 4: ~1 sec
- Cell 5: ~1 sec

**Total:** ~20 seconds to get interactive controls

### Running Inference

1. Select model from dropdown
2. Enter prompt (e.g., "What is 2+2?")
3. Click "▶ Run Reasoning"
4. Watch heatmap update in real-time

---

## Troubleshooting

### "command not found: ./run_jupyter.sh"
```bash
# Make sure script is executable
chmod +x run_jupyter.sh

# Or run with bash explicitly
bash run_jupyter.sh
```

### "externally-managed-environment" error
✅ **Already handled!** The launcher script creates a venv automatically.

### Jupyter takes too long to start
- **First run:** Normal (downloads models)
- **Slow on subsequent runs:** Check disk space
- **Kernel issues:** Restart kernel: Kernel → Restart in menu

### "Connection refused" from remote
```bash
# Check if Jupyter already running
ps aux | grep jupyter

# If stuck, kill it
pkill -f jupyter

# Restart
./run_jupyter.sh
```

### Port already in use
```bash
# Use different port
./run_jupyter.sh 8889

# Or find what's using port 8888
lsof -i :8888
```

---

## File Structure

```
phase6/
├── run_jupyter.sh                           ← Launch script (use this!)
├── interactive_reasoning_tracker.ipynb      ← Notebook file
├── venv/                                    ← Virtual env (auto-created)
│   ├── bin/
│   │   └── python
│   ├── lib/
│   └── ...
└── requirements_phase6.txt                  ← Dependencies
```

---

## Environment Management

### Start/Stop

```bash
# Start (with venv auto-activated)
./run_jupyter.sh

# Stop: Ctrl+C in terminal or close Jupyter
```

### Manual venv activation (if needed)
```bash
source venv/bin/activate
python -m jupyter notebook interactive_reasoning_tracker.ipynb
```

### Clean venv (reset everything)
```bash
rm -rf venv/
./run_jupyter.sh  # Recreates fresh venv
```

---

## Performance Tips

1. **First model load:** 2-3 minutes (downloads from HuggingFace)
2. **Subsequent runs:** 30 seconds (cached)
3. **Per-token inference:** 200-500ms (depends on model size)
4. **GPU required:** Yes (RTX 6000 recommended)
5. **Memory usage:** 2-4 GB VRAM

---

## What Next?

Once you've tried the notebook:

1. **Explore different models** - Notice architectural differences
2. **Try various prompts** - Math, logic, multi-step
3. **Read Phase 6 docs** - Understand layer meanings
4. **Check Phase 5.4 results** - See which SAEs are trained
5. **Plan Phase 6.1** - Feature ablation next

---

## Related Files

- [JUPYTER_NOTEBOOK_GUIDE.md](JUPYTER_NOTEBOOK_GUIDE.md) - Full documentation
- [interactive_reasoning_tracker.ipynb](interactive_reasoning_tracker.ipynb) - The notebook
- [run_tracker.sh](run_tracker.sh) - Flask version launcher (alternative, not needed)

---

✅ **Ready to go! Run: `./run_jupyter.sh` 🚀**
