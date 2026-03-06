# Diabetic retinopathy detection challenge

A [CodaBench](https://www.codabench.org/) machine learning competition for classifying retinal fundus images into **5 severity levels** of diabetic retinopathy.

## Challenge overview

Diabetic retinopathy (DR) is a diabetes complication that affects the eyes and is a leading cause of blindness worldwide. Early detection through automated analysis of retinal images is crucial for timely treatment.

In this challenge, participants must build a model that classifies retinal fundus images into one of the following severity levels:

| Label | Severity Level |
|-------|---------------|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

The task is framed as a **multi-label classification** problem (one-hot encoded labels), evaluated using **F1 Macro**, **F1 Micro**, and **Hamming Loss**.

## Dataset

- **Source**: [IDRiD — Indian Diabetic Retinopathy Image Dataset](https://universe.roboflow.com/officeworkspace/diabetic-retinopathy-dataset) (via Roboflow)
- **License**: CC BY 4.0
- **Total images**: 510 retinal fundus photographs
  - Train: 407 images
  - Test: 103 images (split into public/private for the two competition phases)
- **Data preparation script**: [`tools/setup_data.py`](tools/setup_data.py)

## Repository structure

```
challenge2/
├── competition.yaml          # CodaBench competition configuration
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── dataset/                  # Raw dataset (from Roboflow)
│   ├── train/                # Training images + _classes.csv
│   └── test/                 # Test images + _classes.csv
│
├── dev_phase/                # Phase 1 — Development (public test)
│   ├── input_data/           # train.csv, test.csv, images
│   └── reference_data/       # Ground truth labels
│
├── test_phase/               # Phase 2 — Final (private test)
│   ├── input_data/
│   └── reference_data/
│
├── ingestion_program/        # Ingestion pipeline (runs participant code)
│   └── ingestion.py
│
├── scoring_program/          # Scoring pipeline (computes metrics)
│   └── scoring.py
│
├── solution/                 # Baseline submission
│   └── submission.py         # ResNet-18 fine-tuning baseline
│
├── tools/                    # Utility scripts
│   └── setup_data.py         # Dataset preparation & split script
│
├── pages/                    # CodaBench competition page
│   └── overview.md
│
└── starting_kit.ipynb        # Getting started notebook (EDA + baseline)
```

## Getting started

1. Open the [`starting_kit.ipynb`](starting_kit.ipynb) notebook for:
   - Exploratory Data Analysis (EDA)
   - Understanding the evaluation metrics
   - Running the baseline model

2. Create your own `submission.py` following the interface defined in [`solution/submission.py`](solution/submission.py).

3. Submit your solution on CodaBench.

## Team members

- **Member 1** — Maxence Debes
- **Member 2** — Sinoué Gad
- **Member 3** — Vadim Hemzellec-Davidson
- **Member 4** — Agathe Santus
- **Member 5** — Nathan Vanier de Saint-Aulnay

## References

- [IDRiD Dataset on Roboflow](https://universe.roboflow.com/officeworkspace/diabetic-retinopathy-dataset)
- [CodaBench Documentation](https://codabench.org/)
