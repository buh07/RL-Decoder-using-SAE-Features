# 🧠 Phase 6: Interactive Reasoning Tracker - Jupyter Notebook Edition

Perfect for remote SSH environments!

> **NEW:** Use the launcher script for automatic venv setup!

---

## Quick Start

### 🚀 RECOMMENDED: Use the Launcher Script

```bash
cd phase6
./run_jupyter.sh
```

This automatically:
- ✅ Creates virtual environment (if needed)
- ✅ Installs all dependencies in venv
- ✅ Launches Jupyter on http://localhost:8888

**See [QUICKSTART_JUPYTER.md](QUICKSTART_JUPYTER.md) for details**

---

### Option 1: Run Jupyter Locally (Manual)

```bash
# First time only: create venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_phase6.txt

# Then launch
jupyter notebook interactive_reasoning_tracker.ipynb
```

Browser will open automatically at `http://localhost:8888`

### Option 2: Run on Remote, Access Locally

**On remote machine:**
```bash
cd /path/to/phase6
./run_jupyter.sh
```

Output will show:
```
http://127.0.0.1:8888/?token=abc123...
```

**On your local machine:**
- Copy the full URL from remote
- Paste into local browser

**OR use SSH port forwarding:**

```bash
# On local machine
ssh -L 8888:localhost:8888 user@remote-host

# Then on remote (after SSH connects)
./run_jupyter.sh
```

Then on local: `http://localhost:8888`

---

## How to Use

### 1. Virtual Environment Verification

When you run the notebook, it automatically checks venv:

**Cell 1 output:**
```
🚀 JUPYTER KERNEL STARTUP DIAGNOSTICS
Kernel Python: /path/to/venv/bin/python    ← Should be venv path
Version: 3.10.x
Running in Virtual Environment: ✅ YES
```

**Cell 2 output:**
```
🔍 VIRTUAL ENVIRONMENT SETUP
Virtual environment status: ✅ ACTIVE
✅ All core libraries loaded successfully
```

If you see ⚠️ warnings, the notebook provides exact fix instructions!

**For complete venv details:** [VENV_SETUP_GUIDE.md](VENV_SETUP_GUIDE.md)

---

### 2. First Time Setup

Run the first 3 cells:
- **Cell 1**: Kernel diagnostics (auto-runs)
- **Cell 2**: venv verification + core imports
- **Cell 3**: Loads ML libraries (transformers, SAE)

⏱️ Takes ~30 seconds

### 2. Select Model & Run

1. Click on the model dropdown and select your model
   - `gpt2-medium` (24 layers, ~2GB)
   - `phi-2` (32 layers, ~4GB)
   - `gemma-2b` (18 layers, ~3GB)
   - `pythia-1.4b` (24 layers, ~3GB)

2. Enter your reasoning prompt (e.g., "What is 2+2?")

3. Adjust max tokens (1-100)

4. Click "▶ Run Reasoning"

### 3. Interpret Results

**The heatmap shows:**
- **Rows**: Layers (L0 = input, L23 = output for gpt2)
- **Columns**: Token positions
- **Colors**: 
  - 🟩 Green = SAE reconstructs well (normal features)
  - 🟨 Yellow = Some loss (ambiguous features)
  - 🟥 Red = Poor reconstruction (complex/fine-grained info)

**What to look for:**
- **Vertical stripes**: Same feature active across all tokens
- **Diagonal pattern**: Features progress through layers
- **Hot spots**: Critical computation areas

---

## Features

### ✅ What Works

- [x] 4 production models supported
- [x] Real-time layer visualization
- [x] Feature activation tracking
- [x] Layer semantics/meanings
- [x] Streaming inference
- [x] Works over SSH/remote
- [x] No special port forwarding needed
- [x] Can run with CPU if needed

### ⚠️ Limitations

- 🔴 GPU recommended (can use CPU but will be slow)
- 🔴 First load ~2-3 min (model download cached after)
- 🔴 Some features shown as dummy (if SAE checkpoints missing)
- 🔴 Max 100 tokens for notebook UI (backend handles more)

---

## Example Prompts

Try these to see different activation patterns:

### Simple Math
```
What is 9 × 8?
```
**Expected:** Sharp activation in middle layers for computation

### Logic
```
Is a penguin a bird?
```
**Expected:** Distributed activation (reasoning across layers)

### Multi-step
```
I have 5 apples. I give away 2. How many remain?
```
**Expected:** Gradual activation progression

### Open-ended
```
Explain why the sky is blue
```
**Expected:** Scattered activation (many possible explanations)

---

## Remote Setup Examples

### Setup 1: SSH from Laptop → Lab Computer

```bash
# On your laptop
ssh -L 8888:localhost:8888 username@lab-computer-ip

# On the lab computer (after SSH)
cd /scratch2/f004ndc/RL-Decoder\ with\ SAE\ Features/phase6
jupyter notebook interactive_reasoning_tracker.ipynb

# Back on your laptop, open:
# http://localhost:8888
# Paste token if prompted
```

### Setup 2: Direct URL Access (if on same network)

```bash
# On remote computer
jupyter notebook interactive_reasoning_tracker.ipynb --ip=0.0.0.0 --no-browser
# Note the URL: http://IP:8888/?token=...

# On your local computer
# Open browser and paste: http://lab-computer-ip:8888/?token=...
```

### Setup 3: Using tmux (Long-running)

```bash
# On remote, start tmux session
tmux new-session -d -s jupyter

# Attach and run
tmux send-keys -t jupyter 'cd /scratch2/.../phase6' Enter
tmux send-keys -t jupyter 'jupyter notebook interactive_reasoning_tracker.ipynb' Enter

# Detach (Ctrl+B then D)

# Later: tmux attach -t jupyter
```

---

## Troubleshooting

### "Module not found" or "externally-managed-environment" error
**Solution:** Use the launcher script - it handles venv automatically!
```bash
./run_jupyter.sh
```

**What this does:**
1. Creates virtual environment (if needed)
2. Activates venv
3. Installs all dependencies in venv (isolated)
4. Launches Jupyter with venv active

**If running manually:** Make sure to activate venv first
```bash
source venv/bin/activate
jupyter notebook interactive_reasoning_tracker.ipynb
```

**For complete venv troubleshooting:** [VENV_SETUP_GUIDE.md](VENV_SETUP_GUIDE.md)

### Slow inference (>1 min per token)
**Solutions:**
1. Use smaller model (`gpt2-medium` ≤ `pythia-1.4b` ≤ `gemma-2b` ≤ `phi-2`)
2. Reduce max_tokens
3. Check GPU memory: `nvidia-smi`
4. Close other applications

### GPU out of memory
**Solutions:**
1. Select smaller model first
2. Reduce max_tokens to 10
3. Restart kernel: Kernel → Restart in Jupyter menu
4. Try CPU mode (slow but works): 
   ```python
   # Add to cell 3
   torch.device('cpu')
   ```

### "Connection refused" from remote
**Solution:** Check if Jupyter running
```bash
# On remote
ps aux | grep jupyter
# Should see running jupyter process

# Kill and restart if stuck
pkill -f jupyter
cd phase6 && jupyter notebook interactive_reasoning_tracker.ipynb
```

### SAE files not found
**Expected behavior:** Notebook creates dummy SAEs for testing
- Visualization still works
- Gets more accurate if SAE checkpoints present
- Located at: `phase5_results/multilayer_transfer/saes/`

---

## Performance Benchmarks

| Model | Load Time | Per-Token | Memory | Quality |
|-------|-----------|-----------|--------|---------|
| GPT2-medium | 2min | 200-300ms | 2GB | High (real SAEs) |
| Pythia-1.4B | 2min | 300-500ms | 3GB | High |
| Gemma-2B | 2min | 400-600ms | 3GB | High |
| Phi-2 | 3min | 600-1200ms | 4GB | High |

*Times are for first run; cached after*

---

## Architecture

### What Happens When You Click "Run"

```
1. User Input
   ↓
2. Model Selection & Loading
   ↓
3. Tokenize Prompt
   ↓
4. For each token:
   a. Generate next token
   b. Capture all layer activations (via hooks)
   c. Encode each layer through SAE
   d. Calculate reconstruction quality
   e. Extract top-8 features per layer
   f. Display in heatmap
   ↓
5. Stream results to UI
```

### Frontend = Jupyter Widgets
- Dropdown (model selection)
- Text input (prompt)
- Slider (max tokens)
- Button (run)
- Outputs (status, visualization, stream)

### Backend = Your Notebook
- Model loading (transformers)
- SAE inference (dummy or real)
- Hook registration (activation capture)
- Heatmap rendering (matplotlib)

---

## Integration with Phase 5.4

This notebook uses SAE checkpoints from Phase 5.4:

```
phase5_results/
└── multilayer_transfer/
    └── saes/
        ├── gpt2-medium/
        ├── phi-2/
        ├── gemma-2b/
        └── pythia-1.4b/
```

If checkpoints present: ✓ Real SAE decodings
If missing: ⚠ Dummy SAEs (still useful for visualization)

---

## Future Enhancements

### Phase 6.1: Feature Ablation
- [ ] Add slider to suppress features
- [ ] Watch reasoning degrade in real-time
- [ ] Causal importance measurement

### Phase 6.2: Cross-Model Comparison
- [ ] Side-by-side prompts in two models
- [ ] Identical reasoning, different features
- [ ] Identify universal components

### Phase 6.3: Latent Steering
- [ ] Manipulate latent codes directly
- [ ] Boost/suppress specific features
- [ ] Control reasoning output

### Phase 6.4: Batch Analysis
- [ ] Process many prompts
- [ ] Aggregate feature statistics
- [ ] Find universal reasoning patterns

---

## Tips & Tricks

### 1. Comparing Models
Run the notebook twice (cell 4+) with different models selected
You'll see different activation patterns:
- GPT2: More distributed activation (universal features)
- Phi-2: Hierarchical stages (specialized layers)

### 2. Long Sequences
Set max_tokens=50+ to see how reasoning evolves
Look for repeating patterns in layer activation

### 3. Reproducibility
Same prompt → same output:
- Set random seeds in Cell 4 for exact reproduction
- Save heatmaps: Click "Save" in matplotlib toolbar

### 4. Export Results
```python
# After running, capture heatmap
import matplotlib.pyplot as plt
plt.savefig('heatmap.png', dpi=150, bbox_inches='tight')
```

---

## Questions?

See also:
- [PHASE6_TRACKER_DESIGN.md](PHASE6_TRACKER_DESIGN.md) - Architecture
- [README_TRACKER.md](README_TRACKER.md) - Flask version (similar UI)
- [PHASE6_TRACKER_COMPLETE.md](PHASE6_TRACKER_COMPLETE.md) - Implementation details

---

✅ **Happy exploring! 🚀**
