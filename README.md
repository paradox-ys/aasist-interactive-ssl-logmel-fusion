# SSL-AASIST with Deep log-Mel Interactive Fusion

This repository contains an audio deepfake detection baseline that combines:

- an XLS-R/Wav2Vec2 SSL front-end,
- a deep log-Mel acoustic branch,
- bidirectional interactive attention fusion, and
- an AASIST-style spectro-temporal graph attention back-end.

The implementation is adapted from the SSL anti-spoofing/AASIST family of baselines, with an added log-Mel branch and cross-feature fusion module for ASVspoof-style speech deepfake detection.

## Method

The model uses two complementary feature streams:

1. **SSL branch**: raw waveform is encoded by XLS-R/Wav2Vec2 and projected to 128 dimensions.
2. **log-Mel branch**: waveform is converted to log-Mel features, then processed by multi-scale depthwise temporal convolutions, BiLSTM, and channel gating.
3. **Interactive fusion**: SSL and log-Mel features are aligned with bidirectional cross-attention. The fusion input includes SSL features, log-Mel features, multiplicative consistency cues, and absolute discrepancy cues.
4. **AASIST back-end**: the fused sequence is treated as a time-frequency representation and classified with CNN residual blocks plus spectro-temporal graph attention.

## Repository Layout

```text
.
├── model.py                  # SSL + log-Mel fusion + AASIST-style graph model
├── main_SSL_LA.py            # Training and score generation entry point
├── data_utils_SSL.py         # ASVspoof dataset loading and RawBoost preprocessing
├── RawBoost.py               # RawBoost augmentation
├── eval_metric_LA.py         # LA EER / t-DCF utilities
├── eval_metrics_DF.py        # DF EER utilities
├── evaluate_2021_LA.py       # ASVspoof2021 LA official-style evaluation
├── evaluate_2021_DF.py       # ASVspoof2021 DF official-style evaluation
├── core_scripts/
│   └── startup_config.py
└── database/
    ├── ASVspoof_LA_cm_protocols/
    └── ASVspoof_DF_cm_protocols/
```

Data, official evaluation keys, logs, and model checkpoints are intentionally excluded from this repository.

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

You also need a local XLS-R/Wav2Vec2 model directory or a HuggingFace model name. On AutoDL, the experiments used:

```text
/root/autodl-tmp/xlsr_300m_hf
```

Pass another path with `--ssl_model_path` if needed.

## Data Layout

The scripts expect ASVspoof data in the following style:

```text
/path/to/data/LA/
├── ASVspoof2019_LA_train/flac/*.flac
├── ASVspoof2019_LA_dev/flac/*.flac
└── ASVspoof2019_LA_eval/flac/*.flac    # optional

/path/to/data/
├── ASVspoof2021_LA_eval/flac/*.flac
└── ASVspoof2021_DF_eval/flac/*.flac
```

## Training

Example training command on ASVspoof2019 LA:

```bash
python main_SSL_LA.py \
  --database_path /root/autodl-tmp/data/LA/ \
  --protocols_path database/ \
  --ssl_model_path /root/autodl-tmp/xlsr_300m_hf \
  --batch_size 8 \
  --num_epochs 50 \
  --lr 1e-6 \
  --comment logmel_fusion
```

## ASVspoof2021 LA Evaluation

Generate scores:

```bash
python main_SSL_LA.py \
  --database_path /root/autodl-tmp/data/ \
  --protocols_path database/ \
  --track LA \
  --eval \
  --model_path models/model_LA_weighted_CCE_50_8_1e-06_logmel_fusion/best_model.pth \
  --eval_output scores_2021_LA.txt
```

Compute LA metrics after downloading the official ASVspoof2021 LA keys:

```bash
python evaluate_2021_LA.py scores_2021_LA.txt ./LA-keys-stage-1/keys eval
```

## ASVspoof2021 DF Evaluation

Generate scores:

```bash
python main_SSL_LA.py \
  --database_path /root/autodl-tmp/data/ \
  --protocols_path database/ \
  --track DF \
  --eval \
  --model_path models/model_LA_weighted_CCE_50_8_1e-06_logmel_fusion/best_model.pth \
  --eval_output scores_2021_DF.txt
```

Compute DF metrics after downloading the official ASVspoof2021 DF keys:

```bash
python evaluate_2021_DF.py scores_2021_DF.txt ./DF-keys-stage-1/keys eval
```

## Notes

- This repository does not include ASVspoof audio data, official keys, XLS-R weights, or trained checkpoints.
- Labels follow the common ASVspoof convention used in the original SSL baseline: `bonafide = 1`, `spoof = 0`.
- The model checkpoint paths in the examples are placeholders; train your own model or copy your checkpoint into `models/`.

## Acknowledgements

This code builds on the SSL anti-spoofing and AASIST baseline ecosystem. Please cite the corresponding original works when using this repository for research.
