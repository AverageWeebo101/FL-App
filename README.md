# An Enhanced Federated Cycle for DeepFake Detection

<div align="center">







**A thesis-oriented Python project that combines deepfake video detection with an interactive federated learning defense demo.**

</div>

## Overview

This repository contains two connected parts of the same research project:

1. **Deepfake video detection** through a desktop GUI application that analyzes videos frame by frame.
2. **Enhanced federated learning cycle visualization** through an interactive defense/demo application that explains how secure and quality-aware client participation works.

The study centers on improving the federated learning cycle used for deepfake detection by introducing **dynamic client selection, update validation, contribution weighting, and a client reputation ledger**. According to the uploaded research manuscript, the work uses the **FaceForensics c23** subset, a **70/15/15** train-validation-test split, **10 simulated clients**, and evaluates performance using **accuracy, F1-score, ROC-AUC, and inference latency**.

## Project Highlights

### Deepfake Detector (`deepfake_detector_video.py`)

- Desktop GUI for loading and analyzing video files.
- Face detection using **MediaPipe**.
- TFLite-based deepfake inference on detected faces.
- Real-time overlay showing label, confidence, frame progress, FPS, and inference time.
- Playback controls including play, pause, stop, restart, and speed adjustment.
- Export of frame-level results to **CSV**.
- Save current frame as **PNG**.

### Defense Demo (`defense_demo_v6-3.py`)

- Interactive **Tkinter + Matplotlib** demo for thesis defense or presentation.
- Visual walkthrough of the federated learning round pipeline.
- Step-by-step animation of:
  - Client selection
  - Local training
  - Model update sending
  - Update validation
  - Weighted aggregation
  - Reputation updating
- Built-in round metrics and charts using prepared round data.
- Integrated detector panel for video-based deepfake inspection.

## Research Context

This project as an enhancement of the **FL-TENB4** baseline for deepfake detection. The proposed system improves the original federated learning cycle through four main active mechanisms:

- **Dynamic client selection** using validation quality, data volume, latency, reputation, and staleness.
- **Server-side update validation** to reject or suppress harmful client updates.
- **Contribution-weighted aggregation** instead of plain equal-weight averaging.
- **Client reputation ledger** to track reliability across rounds.

The study simulates a federated environment with **10 clients**, selects **3 clients per round**, and runs for **10 communication rounds**. The manuscript also notes that a **server-side knowledge distillation** module was implemented but disabled in the final run because the available setup was not considered data-rich enough to make the teacher ensemble sufficiently reliable.

## Repository Structure

```text
.
├── defense_demo_v6-3.py         # Demo application
├── deepfake_detector_video.py   # Standalone deepfake video detector GUI
├── requirements.txt             # Python dependencies
├── CommandsToRun.txt            # Quick setup and run guide
└── README.md                    # Project documentation
```

## Requirements

- **Python 3.12** recommended.
- **Tkinter** available in the Python installation.
- A compatible **`.tflite` model file** for inference.

> **Note:** Tkinter is usually bundled with standard Python on Windows. On some Linux distributions, it may need to be installed separately through the system package manager.

## Installation

### 1) Create a virtual environment

```bash
python -m venv deepfake_env
```

### 2) Activate the environment

#### Windows CMD

```bat
deepfake_env\Scripts\activate
```

#### Windows PowerShell

```powershell
.\deepfake_env\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
source deepfake_env/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run

### Run the deepfake detector

```bash
python deepfake_detector_video.py
```

### Run the defense demo

```bash
python defense_demo_v6-3.py
```

### Optional: specify a model manually

```bash
python deepfake_detector_video.py --model path/to/model.tflite
python defense_demo_v6-3.py --model path/to/model.tflite
```

## Expected Workflow

### Detector workflow

1. Launch the detector.
2. Open a supported video file.
3. Let the application detect faces and run frame-level inference.
4. Review live predictions and the overall verdict.
5. Export CSV logs or save frames when needed.

### Defense demo workflow

1. Launch the defense demo.
2. Load the model if prompted.
3. Step through the federated learning cycle visually.
4. Use the charts and status panels during thesis presentation.
5. Optionally test the integrated detector panel with a video.

## Supported Video Formats

The scripts define support for common video formats including:

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.webm`
- `.flv`
- `.wmv`
- `.m4v`

## Technical Notes

- The detector uses **face-based preprocessing** and TFLite inference.
- The demo script includes prepared round metrics and visualization logic for federated training behavior.
- The codebase uses **TensorFlow**, **NumPy**, **MediaPipe**, **OpenCV**, **Pillow**, and **Matplotlib**.
- If the model file is not found automatically, the applications may prompt for its location.

## Research Configuration Summary

The experimental setup includes:

| Item | Details |
|------|---------|
| Dataset | FaceForensics c23 subset |
| Manipulation types | DeepFakes and FaceSwap |
| Split ratio | 70/15/15 |
| Federated clients | 10 simulated clients |
| Images per client | 920 |
| Selected clients per round | 3 |
| Rounds | 10 |
| Local batch size | 32 |
| Local epochs | 5 |
| Metrics | Accuracy, F1-score, ROC-AUC, inference latency |

## Limitations

The current study is limited to:

- The **FaceForensics c23** dataset.
- The **DeepFakes** and **FaceSwap** manipulation categories.
- A **simulated federated environment**.
- Comparison primarily against the **FL-TENB4** baseline.

Because of this, results may not directly generalize to other datasets, deployment settings, or manipulation families.