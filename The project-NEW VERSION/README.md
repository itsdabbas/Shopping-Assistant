# GazeShop: Multimodal Gaze and Speech Assistant

Welcome to the **GazeShop** project. This repository serves as the central hub for our Multimodal Shopping Assistant toolkit. GazeShop aims to revolutionize e-commerce interactions by allowing users to navigate and act on a digital storefront using a seamless combination of **eye-tracking (Gaze)** and **voice commands (Speech)**.

---

## 🏗️ Project Architecture

To maintain a clean and decoupled codebase, the project is divided into distinct modular domains:

1. **`Gaze_Detection-main/` (Visual Modality)**: 
   Handles camera feed processing, media-pipe landmarks extraction, and actual iris/gaze calibration. It detects what item on the screen the user is directly looking at and outputs standard `GazeEvent` data.
   
2. **`Speech_Detection-main/` (Speech Modality)**:
   A fully decoupled toolkit for voice interaction. This includes Push-to-Talk logic, advanced Voice Activity Detection (Silero neural-network VAD), speech-to-text translation (via optimized Whisper/Vosk to prevent hallucinations), intent classification (determining *what* the user meant), and dialogue management (asking for clarifications if needed). It publishes standard `SpeechEvent` data.

3. **`Test_Sandbox/` (Multimodal Integration & E2E Testing)**:
   The ultimate proving ground where both modalities meet. Using a central `EventBus` and our `FusionEngine`, this sandbox syncs visual inputs with speech commands in a time-window to execute context-aware multimodal actions (e.g. looking at a shirt and saying "add this to cart").

---

## 🛠️ Environment Setup & Installation

Follow these steps to set up the project on your local machine. It is highly recommended to use a clean Python virtual environment to prevent package conflicts.

### Step 1: Create a Virtual Environment
Open your terminal (PowerShell or Command Prompt) in the root of the project (`The project` folder) and run:
```bash
python -m venv venv
```

### Step 2: Activate the Virtual Environment
Before installing packages or running the code, activate the virtual environment:
**On Windows:**
```bash
.\venv\Scripts\activate
```
*(On macOS/Linux: `source venv/bin/activate`)*

### Step 3: Install Dependencies
With the virtual environment active, install the required packages for both modalities.

First, install the dependencies for the Speech Modality:
```bash
pip install -r "Speech_Detection-main/requirements.txt"
```

Next, install the dependencies for the Gaze Modality:
```bash
pip install opencv-python mediapipe websockets numpy
```

Finally, install the additional core libraries required for the Multimodal Sandbox integration:
```bash
pip install sounddevice pynput
```

### Step 4: Verify the Setup
Your environment is now fully set up! You can proceed to test the integration by running the live sandbox:
```bash
cd Test_Sandbox
python live_e2e_harness.py --mode live
```

---

## ⚙️ How the Modalities Work Together

The true power of GazeShop lies in its **Late-Fusion Architecture**.

1. **Independent Sensing:** The camera constantly tracks your eyes, locking onto "virtual shelves" when a sustained gaze is detected. This creates an ongoing stream of `LOCK` and `UNLOCK` events on our `EventBus`.
2. **Intent Recognition:** When you hold the microphone key and speak, a multi-tiered VAD (Voice Activity Detection) trims silence, and an optimized ASR engine parses the sentence. Regex patterns then determine your intent (e.g., `ADD_TO_CART`, `SCROLL_DOWN`). The system will also switch to "dialogue-active" mode to capture yes/no confirmation commands flawlessly.
3. **Time-Window Fusion:** The `FusionEngine` actively listens to both streams. If it receives a speech command that requires a visual target (like "buy *this*"), it checks the recent history (within a 3.0-second window) to see what target the `GazeEvent` had locked onto. 
4. **Dialogue Recovery:** If you speak an object-bound command *without* looking at a product, or if your gaze was ambiguous between two items, the `DialogueManager` steps in automatically to ask you for clarification (e.g., "Which one did you mean? Left or right?").

---

## 🧪 Live E2E Harness (Test Sandbox)

The `Test_Sandbox` allows developers to test the full multimodal integration without needing to hook up a completely styled Web Interface immediately. It provides a visual OpenCV window overlaid with "Virtual Shelves" to test Gaze tracking capability, while actively listening to the microphone for commands.

### Running the Live Multimodal Test

**Prerequisites:** You must have installed the dependencies from both the Gaze and Speech modules, primarily `mediapipe`, `opencv-python`, `sounddevice`, `pynput`, and your chosen ASR engine.

To start the sandbox in interactive Live mode:

```bash
cd Test_Sandbox
python live_e2e_harness.py --mode live
```

**How to interact:**
1. A camera window will open showing your face, with multiple colored rectangles representing virtual items (`item_1`, `item_2`, etc.).
2. **Aim the on-screen cursor** towards one of the boxes to trigger a `Gaze Lock`.
3. Press and hold the **'m' key** (on the OpenCV window) to activate the microphone. Speak a command (e.g., "Add this to cart"), then release the key.
4. Watch the console or the on-screen logs to see the system successfully merge your visual target and speech intent!

You can also run automated, deterministic tests to verify the logic of the `FusionEngine` without a camera using:
```bash
python live_e2e_harness.py --mode scripted
```

---

## ⚠️ Important Note Regarding Live Visual Tracking (Nose vs Gaze)

If you run the Live Test Sandbox, you might notice something completely valid but counter-intuitive: **the visual cursor tracks the center of your face (around your nose) rather than your actual irises / gaze vectors.**

### Why does the Live Test use the Nose instead of real Gaze?

1. **Calibration Dependency:** True iris-to-screen coordinate mapping requires a dedicated 9-point user calibration phase mapping the physical screen dimensions to eye geometry. Our full UI component (`Gaze_Detection-main`) executes this calibration.
2. **Rapid Sandbox Testing:** The primary goal of the `live_e2e_harness` is to test the **Multimodal Fusion Logic** (i.e. verifying that "Look at A" + "Say B" merges successfully) in an instant-startup developer environment, *not* to test the accuracy of the eye-tracking model itself.
3. **Stability & Reliability:** By skipping the tedious 15-second calibration phase every time you run the script, we mathematically fallback to the center of the `face_box` bounding box provided by MediaPipe. The nose/center-face provides a highly stable, deterministic coordinate that developers can easily point at virtual on-screen boxes by slightly moving their head. 

Rest assured, **when integrated into the final consumer web interface**, the system correctly subscribes to the fully calibrated Iris Landmarker to guarantee true Eye-Gaze interaction. For now, enjoy testing the integration seamlessly using head movements!
