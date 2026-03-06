# Sarcasm Detection Module

## Overview

This module implements the sarcasm detection component of the cyberbullying detection system. Sarcasm can obscure abusive intent by presenting negative statements in a seemingly positive tone. Detecting sarcasm helps improve the overall accuracy of bullying detection.

The model implemented in this module is based on a **Bidirectional Gated Recurrent Unit (BiGRU) with an attention mechanism**, designed to capture contextual cues and subtle linguistic signals commonly associated with sarcasm.

## Model Architecture

The sarcasm detection model follows the architecture below:

Input Text
→ Character Tokenization
→ Embedding Layer
→ BiGRU (Bidirectional GRU)
→ Attention Layer
→ Fully Connected Layer
→ Sigmoid Output

### Key Components

**BiGRU Layer**

* Processes text in both forward and backward directions
* Captures contextual dependencies across the sentence

**Attention Mechanism**

* Assigns higher importance to words or symbols that indicate sarcasm
* Helps the model focus on critical cues such as emojis or unusual phrasing

**Sigmoid Output**

* Produces a probability score between 0 and 1 indicating sarcasm likelihood

## Training

The model is trained using sarcasm datasets collected from Twitter and Reddit. The objective is binary classification:

* **1** → Sarcastic
* **0** → Non-sarcastic

Training includes:

* Character-level tokenization
* Cross-entropy loss optimization
* Early stopping to prevent overfitting
* Model checkpoint saving

## Model Output

For each input message, the model produces:

* **P_sar (Sarcasm Probability)**

Example:

Input:
"Wow you're such a genius 🤡"

Output:

Sarcasm Probability: 0.81

## Role in the Cyberbullying System

The sarcasm model is not used alone to determine bullying. Instead, it provides contextual information that is combined with other models in the **fusion layer**.

The fusion layer combines:

* **P_cb** → Cyberbullying probability (MTKD XGBoost)
* **P_sar** → Sarcasm probability
* **P_emo** → Emotion probability

These signals together produce the final cyberbullying decision.

## Files in This Module

| File          | Description                           |
| ------------- | ------------------------------------- |
| train.py      | Training script for the sarcasm model |
| dataset.py    | Dataset loading and preprocessing     |
| model.py      | BiGRU + attention architecture        |
| evaluate.py   | Model evaluation script               |
| best_model.pt | Saved trained model checkpoint        |

## Integration

During inference, the sarcasm model receives text input and outputs a sarcasm probability. This value is later used by the system's fusion layer to improve cyberbullying detection accuracy.
