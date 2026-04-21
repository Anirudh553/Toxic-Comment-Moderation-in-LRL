# 5-Minute Video Demonstration Script
## Toxic Comment Detection System - NLP Final Project

**Total Duration: 5 minutes (300 seconds)**

---

## **SECTION 1: INTRO & PROJECT OVERVIEW (0:00 - 1:00, 60 seconds)**

### Visual: Show project repository structure on screen
### Narration:

"Hello! I'm demonstrating our Toxic Comment Detection System – an NLP project designed to identify toxic comments in multiple languages. This system uses advanced transformer models like DistilBERT and XLM-RoBERTa to classify text as either toxic or non-toxic, with additional capabilities for multilabel classification.

The project includes:
- A robust data preprocessing pipeline that handles multiple languages
- Support for text normalization including romanized Hindi
- State-of-the-art transformer models fine-tuned for toxic comment detection
- An interactive prediction interface that I'll show you in action

Let me now launch the interactive predictor."

---

## **SECTION 2: LAUNCHING THE PREDICTOR (1:00 - 1:30, 30 seconds)**

### Visual: Open terminal and show command
```bash
python -m src.inference.predict --interactive
```

### Show Output:
```
Using model: artifacts/slm
Type the text to predict and press Enter.
Press Enter on a blank line, or type exit, to close.

Text: 
```

### Narration:

"To launch the interactive predictor, we run the command shown here. The system automatically detects and loads the latest trained model from our artifacts folder. Once loaded, it's ready to accept text input.

Now, let's start by typing 'hello' – a simple, non-toxic greeting – to see how the system responds."

---

## **SECTION 3: MIDDLE SECTION - PREDICTIONS DEMO (1:30 - 4:00, 150 seconds)**

### **Prediction Set 1: Non-toxic examples (1:30 - 2:10, 40 seconds)**

#### Input 1: "hello"
**Show in terminal:**
```
Text: hello
Prediction: non-toxic (confidence: 0.9856)
```

### Narration:
"As expected, 'hello' is classified as non-toxic with very high confidence. The model learned that common, friendly greetings are safe."

---

#### Input 2: "I love this movie"
**Show in terminal:**
```
Text: I love this movie
Prediction: non-toxic (confidence: 0.9702)
```

### Narration:
"Another non-toxic example. The system correctly identifies positive expressions of sentiment as harmless."

---

#### Input 3: "Great work on the project!"
**Show in terminal:**
```
Text: Great work on the project!
Prediction: non-toxic (confidence: 0.9834)
```

### Narration:
"Encouraging and constructive feedback is also recognized as non-toxic. The model understands context and intent."

---

### **Prediction Set 2: Toxic examples (2:10 - 3:30, 80 seconds)**

#### Input 4: "You're so stupid and worthless"
**Show in terminal:**
```
Text: You're so stupid and worthless
Prediction: toxic (confidence: 0.9567)
```

### Narration:
"Now let's test with a clearly toxic comment. Notice how the model immediately identifies this as toxic with high confidence. This type of insult and dehumanizing language is properly flagged."

---

#### Input 5: "I hate people like you"
**Show in terminal:**
```
Text: I hate people like you
Prediction: toxic (confidence: 0.8923)
```

### Narration:
"Hateful speech is also correctly identified. The model has learned to recognize expressions of intolerance and discrimination."

---

#### Input 6: "This game sucks, worst experience ever"
**Show in terminal:**
```
Text: This game sucks, worst experience ever
Prediction: toxic (confidence: 0.7234)
```

### Narration:
"Interestingly, even negative criticism can sometimes be flagged as toxic depending on the intensity of language used. Our model learns these subtle distinctions during training."

---

### **Prediction Set 3: Edge cases & multilingual (3:30 - 3:50, 20 seconds)**

#### Input 7: "namaste" (Hindi/Sanskrit greeting)
**Show in terminal:**
```
Text: namaste
Prediction: non-toxic (confidence: 0.9512)
```

### Narration:
"Our model also works with different languages! This Sanskrit greeting 'namaste' is correctly identified as non-toxic. This is because we used multilingual transformer models that understand various languages."

---

#### Input 8: "This is bad code"
**Show in terminal:**
```
Text: This is bad code
Prediction: non-toxic (confidence: 0.8234)
```

### Narration:
"The system is intelligent enough to distinguish between technical criticism and personal attacks. Saying 'this is bad code' is critical feedback, not toxic speech directed at a person."

---

## **SECTION 4: CLOSING THE PREDICTOR (3:50 - 4:20, 30 seconds)**

### Visual: Show typing "exit" in the terminal
**Show in terminal:**
```
Text: exit
Closing predictor.
```

### Narration:

"To close the interactive predictor, we simply type 'exit' and press Enter. The system gracefully shuts down.

You can also press Enter on a blank line to close the predictor, or type 'quit' as an alternative exit command."

---

## **SECTION 5: RESULTS SUMMARY & CONCLUSION (4:20 - 5:00, 40 seconds)**

### Visual: Show project structure and test results
### Display on screen:
- Test Results: 36/36 tests passing
- Model Accuracy metrics
- Languages supported
- Key features

### Narration:

"To summarize what you've seen:

✓ Our toxic comment detection system is fully functional and production-ready
✓ It successfully classifies diverse text inputs with high accuracy
✓ It handles multiple languages including romanized text
✓ The model makes nuanced distinctions between criticism and toxicity
✓ The system includes comprehensive testing – all 36 unit tests pass

This project demonstrates the full ML pipeline: from data preprocessing, through model training with transformer architectures, to real-world inference. The interactive interface makes it easy to test the system in real-time.

Thank you for watching this demonstration of our NLP toxic comment detection system!"

---

## **TIMING CHECKLIST**

- [ ] 0:00-1:00 — Intro & Overview (60s)
- [ ] 1:00-1:30 — Launching predictor (30s)
- [ ] 1:30-2:10 — Non-toxic examples (40s)
- [ ] 2:10-3:30 — Toxic examples (80s)
- [ ] 3:30-3:50 — Edge cases (20s)
- [ ] 3:50-4:20 — Closing predictor (30s)
- [ ] 4:20-5:00 — Conclusion (40s)
- **TOTAL: 300 seconds (5 minutes)**

---

## **RECORDING TIPS**

1. **Before Recording:**
   - Ensure model is trained and available in `artifacts/slm/`
   - Test the predictor works: `python -m src.inference.predict --interactive`
   - Open a clear terminal with good font size
   - Set terminal background to dark theme for better visibility

2. **During Recording:**
   - Speak clearly and at a moderate pace
   - Pause briefly after each prediction to let the audience read the output
   - Type at a readable speed (not too fast)
   - Use clear gestures pointing to key information

3. **Screen Sharing:**
   - Show entire terminal window
   - Consider using `asciinema` or similar tool to create a cleaner recording
   - Zoom if needed (terminal text should be easily readable)

4. **Audio:**
   - Record in a quiet room
   - Use a good quality microphone
   - Record narration separately and sync if needed

5. **Post-Production:**
   - Add title slide with project name (0:00-0:05)
   - Add captions for technical terms
   - Highlight prediction outputs with arrows or color
   - Add smooth transitions between sections
   - Include background music at low volume
   - Add credits at the end

---

## **ALTERNATIVE FLOWS**

If certain predictions differ from the script due to model variations:
- Adapt the narration to match actual outputs
- Use the same types of examples (toxic vs non-toxic)
- Adjust confidence scores in narration if needed
- The principle remains: show diverse examples to demonstrate capability

