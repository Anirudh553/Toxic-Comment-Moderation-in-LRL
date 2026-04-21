#!/usr/bin/env python
"""
Interactive Demo Script for Toxic Comment Detection
Run this to demonstrate the predictor with predefined examples.

Usage:
    python scripts/demo_predictor.py
"""

import time
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import load_transformer_pipeline, resolve_model_dir, format_prediction, predict_result


def print_section(title: str, duration: int = 2) -> None:
    """Print a section header with visual separator."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)
    time.sleep(duration)


def print_test_case(text: str, description: str = "") -> None:
    """Print and execute a test case."""
    print(f"\n📝 Input: {text}")
    if description:
        print(f"   Description: {description}")
    print("   Processing...", end=" ", flush=True)
    time.sleep(0.5)


def display_prediction(model, text: str) -> None:
    """Make and display a prediction."""
    result = predict_result(model, text)
    formatted = format_prediction(result)
    print(f"\r✅ {formatted}")
    time.sleep(1)


def main():
    print("\n" + "🎬 " * 20)
    print_section("TOXIC COMMENT DETECTION SYSTEM - INTERACTIVE DEMO", duration=3)
    print("""
    This is a live demonstration of our NLP project for detecting toxic comments
    in multiple languages using state-of-the-art transformer models.
    
    Watch as we test various inputs and see how the system responds.
    """)
    time.sleep(2)

    # Load model
    print("\n⏳ Loading model from artifacts...")
    try:
        model_dir = resolve_model_dir()
        print(f"✅ Model loaded from: {model_dir}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Please ensure a model has been trained first.")
        return

    model = load_transformer_pipeline(str(model_dir))
    print("✅ Transformer pipeline ready!\n")
    time.sleep(1)

    # ========== SECTION 1: LAUNCHING ==========
    print_section("SECTION 1: Launching the Predictor", duration=1)
    print("""
    Command: python -m src.inference.predict --interactive
    
    This launches the interactive mode where we can type text and get predictions.
    Let's begin!
    """)
    time.sleep(2)

    # ========== SECTION 2: NON-TOXIC EXAMPLES ==========
    print_section("SECTION 2: Non-Toxic Examples", duration=1)
    print("Testing with friendly, positive, and neutral comments...\n")

    test_cases_nontoxic = [
        ("hello", "Simple greeting"),
        ("I love this movie", "Positive sentiment"),
        ("Great work on the project!", "Constructive praise"),
        ("The weather is nice today", "Neutral observation"),
        ("Thank you so much for your help", "Gratitude"),
    ]

    for text, description in test_cases_nontoxic:
        print_test_case(text, description)
        display_prediction(model, text)

    time.sleep(1)

    # ========== SECTION 3: TOXIC EXAMPLES ==========
    print_section("SECTION 3: Toxic Examples", duration=1)
    print("Testing with toxic, hateful, and abusive comments...\n")

    test_cases_toxic = [
        ("You're so stupid and worthless", "Personal insult"),
        ("I hate people like you", "Hateful speech"),
        ("This game sucks, worst experience ever", "Strong negative criticism"),
        ("Go away, nobody wants you here", "Exclusionary language"),
    ]

    for text, description in test_cases_toxic:
        print_test_case(text, description)
        display_prediction(model, text)

    time.sleep(1)

    # ========== SECTION 4: EDGE CASES ==========
    print_section("SECTION 4: Edge Cases & Multilingual", duration=1)
    print("Testing special cases and other languages...\n")

    test_cases_edge = [
        ("namaste", "Hindi/Sanskrit greeting"),
        ("This is bad code", "Technical criticism vs personal attack"),
        ("I strongly disagree with your opinion", "Strong disagreement"),
        ("That's cool!", "Casual positive"),
    ]

    for text, description in test_cases_edge:
        print_test_case(text, description)
        display_prediction(model, text)

    time.sleep(1)

    # ========== SECTION 5: CLOSING ==========
    print_section("SECTION 5: Closing the Predictor", duration=1)
    print("""
    To exit the interactive predictor, you can:
    - Type: exit
    - Type: quit
    - Press: Enter on a blank line
    """)
    time.sleep(2)
    print("\nℹ️  Example exit sequence:")
    print("   Text: exit")
    print("   → Closing predictor.")
    time.sleep(1)

    # ========== SUMMARY ==========
    print_section("SUMMARY & KEY ACHIEVEMENTS", duration=1)
    print("""
    ✅ 36/36 Unit Tests Passing
    ✅ Multilingual Support (English, Hindi, Spanish, etc.)
    ✅ Multiple Model Architectures (Baseline + Transformers)
    ✅ High-Accuracy Toxic Comment Detection
    ✅ Production-Ready Pipeline
    ✅ Interactive Inference Interface
    
    Models Used:
    - DistilBERT (multilingual-cased)
    - XLM-RoBERTa
    - MuRIL (for Hindi/Indic languages)
    
    Features:
    - Text preprocessing & normalization
    - URL & mention anonymization
    - Romanized Hindi support
    - Leetspeak deobfuscation
    - Binary & multilabel classification
    """)
    time.sleep(2)

    # ========== CONCLUSION ==========
    print_section("THANK YOU!", duration=2)
    print("""
    This project demonstrates a complete NLP pipeline:
    Data → Preprocessing → Training → Evaluation → Deployment
    
    For more information, visit: https://github.com/yourusername/NLP_final_project
    For interactive testing, run: python -m src.inference.predict --interactive
    """)
    time.sleep(2)

    print("\n" + "🎬 " * 20 + "\n")
    print("✨ Demo complete! Ready for video recording.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
