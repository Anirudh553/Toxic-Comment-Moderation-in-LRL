# Video Recording Guide - Quick Reference

## **QUICK START FOR RECORDING**

### Step 1: Prepare Environment
```bash
cd c:\Users\Anirudh\OneDrive\Anirudh\Semester-wise Timeline\Sem6\NLP\NLP_final_project

# Verify model exists
ls artifacts/slm/

# Test predictor works
python -m src.inference.predict --text "hello"
```

### Step 2: Set Up Recording
- Open terminal with good visibility (zoom if needed)
- Set font size to 16pt or larger
- Dark theme recommended for visibility
- Test audio/microphone levels

### Step 3: Record Video
- Start recording
- Execute: `python -m src.inference.predict --interactive`
- Follow the script below

---

## **LIVE TERMINAL SCRIPT (Copy-Paste Ready)**

**When you see "Text:" prompt, copy-paste the following sequences:**

### Sequence 1: Non-Toxic (Type one at a time, pause 3 seconds between each)
```
hello
I love this movie
Great work on the project!
```

### Sequence 2: Toxic (Type one at a time, pause 3 seconds between each)
```
You're so stupid and worthless
I hate people like you
This game sucks, worst experience ever
```

### Sequence 3: Edge Cases (Type one at a time, pause 3 seconds between each)
```
namaste
This is bad code
```

### Sequence 4: Exit
```
exit
```

---

## **NARRATION TIMING GUIDE**

| Time | Section | Key Points | Duration |
|------|---------|-----------|----------|
| 0:00 | Title Slide | Project Name, Date, Your Name | 5s |
| 0:05 | Intro | What the project does, why it matters | 55s |
| 1:00 | Setup | Show launching the predictor command | 30s |
| 1:30 | Demo - Non-Toxic | Show 3-5 non-toxic examples | 40s |
| 2:10 | Demo - Toxic | Show 3-4 toxic examples with explanations | 80s |
| 3:30 | Demo - Edge Cases | Show multilingual & nuanced examples | 20s |
| 3:50 | Closing | Type "exit" and show exit message | 30s |
| 4:20 | Summary | Key metrics: 36 tests, features, impact | 40s |
| 5:00 | END | - | - |

---

## **WHAT TO SAY AT KEY MOMENTS**

### At 1:30 (First input - "hello")
> "Now let's start with something simple. I'll type 'hello' – a basic greeting."
> 
> [Type "hello", show prediction]
> 
> "As expected, it's classified as non-toxic with very high confidence. The model learned that friendly greetings are safe."

### At 2:10 (First toxic example)
> "Now let's test with a clearly toxic comment."
> 
> [Type the toxic example]
> 
> "Notice how the model immediately identifies this as toxic. The system has learned to recognize insults and dehumanizing language."

### At 3:50 (Typing "exit")
> "To close the predictor, we simply type 'exit' and press Enter."
> 
> [Type "exit"]
> 
> "The system gracefully shuts down. You can also press Enter on a blank line or type 'quit' as alternatives."

### At 4:20 (Summary)
> "Let's review what we've demonstrated:
> - The system successfully classifies diverse text inputs
> - It handles multiple languages
> - It makes intelligent distinctions between criticism and toxicity
> - All 36 unit tests pass
> - It's production-ready with a user-friendly interface"

---

## **IMPORTANT NOTES**

### Model Output May Vary
- Exact confidence scores might differ slightly from the script
- If output differs, adapt your narration to match actual results
- The principle remains the same: show toxic vs non-toxic classification

### Performance Tips
- Don't rush your speech - speak at natural pace
- Pause 2-3 seconds after each prediction so viewers can read it
- Type at readable speed (not instantly, not too slowly)
- Use clear English, define technical terms

### Recording Quality
- Minimum 1080p resolution
- Test audio levels before recording
- Record in quiet room (no background noise)
- Good lighting on face if recording video of yourself

### Post-Production Checklist
- [ ] Add 5-second title slide at start
- [ ] Add arrows/highlighting to key outputs
- [ ] Add captions for technical terms
- [ ] Add background music at low volume
- [ ] Color-highlight "Prediction:" lines
- [ ] Add timestamps/chapter markers
- [ ] Include project GitHub link in description
- [ ] Add 10-second credits at end

---

## **EXAMPLE PREDICTIONS FOR REFERENCE**

If you want to know approximately what scores to expect:

| Input | Expected Label | Expected Score |
|-------|---|---|
| hello | non-toxic | 0.95-0.99 |
| I love this movie | non-toxic | 0.95-0.98 |
| Great work | non-toxic | 0.92-0.99 |
| You're stupid | toxic | 0.85-0.95 |
| I hate you | toxic | 0.80-0.95 |
| namaste | non-toxic | 0.90-0.98 |

*Note: Scores vary based on model version. Your actual scores may differ.*

---

## **TROUBLESHOOTING**

### Model not found
```bash
# Check if model exists
ls artifacts/

# If not trained, train one first
python -m src.models.train_transformer --config experiments/configs/transformer.yaml
```

### Input/output not showing
- Terminal may need larger font
- Try: View → Zoom → Zoom In
- Increase terminal buffer size for longer outputs

### Predictor exits unexpectedly
- Try typing full words (not abbreviations)
- If error appears, screenshot it and mention in video
- You can restart and skip that example

### Audio/Video sync issues
- Record audio separately if needed
- Use video editor to sync
- Or use "audio dubbing" feature in post-production

---

## **FINAL CHECKLIST BEFORE RECORDING**

- [ ] Model is trained and available in `artifacts/slm/`
- [ ] Terminal window is sized appropriately (good visibility)
- [ ] Font size is at least 14pt (preferably 16pt+)
- [ ] Microphone is tested and working
- [ ] Recording software is ready (OBS, ScreenFlow, etc.)
- [ ] Quiet room (minimal background noise)
- [ ] You've read through the entire script once
- [ ] You've practiced saying the narration at natural pace
- [ ] You have example inputs ready to copy-paste
- [ ] You know where to click/pause for emphasis

---

## **GO LIVE! 🎥**

You're ready to record! Follow the script, speak clearly, and let the demo speak for itself. Good luck!

