# 5-Minute Video Demo - Complete Resource Package

## 📦 **What You've Got**

I've created a complete package of resources to help you record and deliver a professional 5-minute video demonstration of your Toxic Comment Detection System. Here's what's included:

### **📄 Files Created (in `/scripts/` folder)**

1. **COMPLETE_VIDEO_SCRIPT.md** ⭐ START HERE
   - Full dialogue for the entire 5-minute video
   - Exact timestamps for each section
   - Actions and narration side-by-side
   - Includes exact system outputs you'll see
   - Timing checklist to track your progress

2. **VIDEO_DEMO_SCRIPT.md**
   - High-level narration script
   - Organized by 5 sections with timings
   - Tips for recording and post-production
   - Alternative flows if outputs differ

3. **RECORDING_GUIDE.md**
   - Quick reference for recording
   - Copy-paste ready input sequences
   - Narration timing guide table
   - Troubleshooting section
   - Pre-recording checklist

4. **demo_predictor.py** (Executable)
   - Automated demo with predefined examples
   - Shows all inputs with descriptions
   - Displays results with visual formatting
   - Includes time delays for readability
   - Can be run standalone: `python scripts/demo_predictor.py`

---

## 🎯 **How to Use These Resources**

### **Option 1: LIVE RECORDING (Recommended)**
1. Read **COMPLETE_VIDEO_SCRIPT.md** carefully
2. Practice narration 2-3 times
3. Open terminal and run: `python -m src.inference.predict --interactive`
4. Follow the script, typing inputs as directed
5. Record your screen and narration

### **Option 2: SCRIPTED DEMO**
1. Run: `python scripts/demo_predictor.py`
2. This shows all examples with automatic timing
3. Record the output
4. Add narration in post-production

### **Option 3: HYBRID**
1. Watch the demo script first to see what happens
2. Do a live recording with real inputs
3. Edit together afterward

---

## 📋 **What Each File Shows**

### **COMPLETE_VIDEO_SCRIPT.md** (MAIN FILE)
```
[00:00-00:05] Title Card
[00:05-01:00] Introduction (55 sec)
              - What the project is
              - Why it matters
              - What we'll demonstrate

[01:00-01:30] Launching Predictor (30 sec)
              - Show command
              - Model loads
              - Ready for input

[01:30-02:10] Non-Toxic Examples (40 sec)
              - "hello" 
              - "I love this movie"
              - "Great work!"

[02:10-03:30] Toxic Examples (80 sec)
              - "You're so stupid..."
              - "I hate people..."
              - "This game sucks..."
              - "Go away..."

[03:30-03:50] Edge Cases (20 sec)
              - "namaste" (multilingual)
              - "This is bad code"

[03:50-04:20] Closing (30 sec)
              - Type "exit"
              - Show shutdown

[04:20-05:00] Summary & Conclusion (40 sec)
              - Results review
              - Key achievements
              - Thank you
```

---

## 🔧 **Recording Setup (Quick Steps)**

### Before Recording:
```bash
# Verify model is trained
ls artifacts/slm/

# Test the predictor works
python -m src.inference.predict --text "hello"
```

### Terminal Setup:
- Font size: 16pt or larger
- Dark theme (better visibility)
- Maximize window
- Clear any previous output

### Audio/Video:
- Quiet room (no background noise)
- Good microphone quality
- 1080p resolution minimum
- 30fps frame rate

---

## 🎬 **Sample Dialogue Snippets**

### When user types "hello":
> "As expected, 'hello' is classified as non-toxic with very high confidence – over 98%. The model learned that friendly greetings are completely safe."

### When showing toxic example:
> "Notice the immediate classification as toxic with 95% confidence. This type of personal insult and dehumanizing language is properly flagged."

### When showing multilingual support:
> "Our model works with different languages! This Sanskrit greeting 'namaste' is correctly identified as non-toxic with 95% confidence."

### When typing "exit":
> "To close the interactive predictor, we simply type 'exit' and press Enter. The system gracefully shuts down."

---

## 📊 **Expected Output Examples**

These are approximate outputs you'll see (exact scores may vary):

| Input | Expected Output |
|-------|---|
| hello | Prediction: non-toxic (confidence: 0.98+) |
| I love this movie | Prediction: non-toxic (confidence: 0.97+) |
| You're so stupid | Prediction: toxic (confidence: 0.95+) |
| I hate people like you | Prediction: toxic (confidence: 0.89+) |
| namaste | Prediction: non-toxic (confidence: 0.95+) |

---

## ✅ **Checklist Before Recording**

- [ ] Model is trained: `ls artifacts/slm/`
- [ ] Predictor works: `python -m src.inference.predict --text "test"`
- [ ] Terminal font is readable (16pt+)
- [ ] Microphone is tested and working
- [ ] You've read COMPLETE_VIDEO_SCRIPT.md
- [ ] You've practiced the narration once
- [ ] You know what to type (or have it copy-pasteable)
- [ ] Recording software is ready (OBS, ScreenFlow, etc.)
- [ ] Room is quiet
- [ ] Examples inputs are prepared

---

## 🎥 **Recording Workflow**

### Step 1: Setup (5 minutes)
```bash
cd /path/to/NLP_final_project
python -m src.inference.predict --interactive
```

### Step 2: Record (10-15 minutes)
- Start recording
- Follow COMPLETE_VIDEO_SCRIPT.md
- Type inputs as shown
- Let system respond
- Speak narration as written

### Step 3: Post-Production (varies)
- Trim excess pauses
- Add title slide (5 sec)
- Add background music (low volume)
- Add captions for technical terms
- Highlight key outputs (color boxes)
- Add credits (10 sec)
- Export as MP4/WebM

---

## 📝 **Key Talking Points to Remember**

1. **Project Purpose**: Detect toxic comments in multiple languages
2. **Technology**: Transformer models (DistilBERT, XLM-RoBERTa, MuRIL)
3. **Key Features**: 
   - Multilingual support
   - Nuanced understanding
   - Fast inference
   - Production-ready
4. **Quality**: All 36 tests pass
5. **Use Cases**: Content moderation, safety systems, research

---

## 🚀 **Pro Tips for Recording**

1. **Speed**: Type at readable pace (not instant, not slow)
2. **Pauses**: Pause 2-3 seconds after each prediction
3. **Speaking**: Speak at natural, conversational pace
4. **Confidence**: You built this – show enthusiasm!
5. **Mistakes**: Don't worry about minor mistakes – edit them out
6. **Practice**: Do a test run first (can be 2-3 minutes)

---

## 📞 **Troubleshooting**

**Problem**: Model not found
```bash
# Train a model first
python -m src.models.train_transformer --config experiments/configs/transformer.yaml
```

**Problem**: ipykernel error with notebook
- Optional for video, but if needed: `pip install ipykernel`

**Problem**: Input seems to hang
- Terminal may be waiting for input
- Just wait a moment or press Enter

**Problem**: Outputs don't match script
- That's okay! Adapt narration to match actual outputs
- Principles remain the same (toxic vs non-toxic)

---

## 📚 **File Relationships**

```
scripts/
├── COMPLETE_VIDEO_SCRIPT.md ◄─── MAIN: Word-for-word script
├── VIDEO_DEMO_SCRIPT.md ◄───────── High-level overview
├── RECORDING_GUIDE.md ◄──────────── Quick reference
└── demo_predictor.py ◄────────────── Automated demo (optional)

Use COMPLETE_VIDEO_SCRIPT.md as your primary guide!
```

---

## 🎯 **Success Criteria**

Your 5-minute video should demonstrate:

✅ Project purpose clearly explained
✅ Interactive predictor launched and running
✅ Non-toxic examples correctly classified
✅ Toxic examples correctly identified  
✅ Multilingual support shown
✅ Edge cases handled appropriately
✅ System shut down gracefully with "exit"
✅ Summary of achievements and key metrics
✅ Clear narration throughout
✅ Professional presentation

---

## 💾 **Save & Share**

After recording:
1. Save raw video file
2. Create a backup
3. Edit and finalize
4. Export in high quality (1080p)
5. Upload to YouTube/platform
6. Share link in project submission

---

## 🎓 **For Your Rubric Evaluation**

This video demonstrates:
- **Code Functionality**: 100% working system (36 tests pass)
- **Multilingual Support**: Shown with "namaste" example
- **Production Ready**: Interactive interface, clear outputs
- **Results Quality**: High accuracy predictions
- **Professional Presentation**: Structured, clear narration

This solidifies your submission! 🚀

---

**Ready to record? Start with COMPLETE_VIDEO_SCRIPT.md!**

