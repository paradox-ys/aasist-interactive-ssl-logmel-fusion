# Interactive SSL/log-Mel Fusion for Audio Deepfake Detection

> AASIST-style speech anti-spoofing with XLS-R/Wav2Vec2 SSL features, deep log-Mel acoustic features, and interactive feature fusion.

本仓库提供一个面向 **audio deepfake detection / speech anti-spoofing** 的 PyTorch 研究实现。模型沿用 AASIST 的谱-时图注意力后端，将 XLS-R/Wav2Vec2 自监督语音表征与深层 log-Mel 声学表征进行交互式融合，用于 ASVspoof 2019/2021 场景下的 bonafide/spoof 检测。

## Highlights

- **AASIST-style backend**: 使用谱图、时序图和异构图注意力建模时域/频域伪造线索。
- **SSL speech branch**: 使用 XLS-R/Wav2Vec2 提取帧级高层语音表征。
- **Deep log-Mel acoustic branch**: 使用 log-Mel、temporal convolution、BiLSTM 和 channel gate 补充局部声学纹理。
- **Interactive fusion**: 通过双向 cross-attention、乘积项、差异项和逐帧门控建模两类特征之间的一致性与不一致性。
- **ASVspoof evaluation**: 提供 ASVspoof2019 LA 训练，以及 ASVspoof2021 LA/DF score generation 和 metric scripts。

## Experimental Results

The models were trained on ASVspoof2019 LA train, selected on ASVspoof2019 LA dev, and evaluated on ASVspoof2021 LA eval. Lower values are better.

| Model variant | ASVspoof2019 LA dev EER | ASVspoof2021 LA EER | ASVspoof2021 LA min t-DCF | Note |
|---|---:|---:|---:|---|
| log-Mel fusion | 0.579153% | 2.28% | 0.2388 | SSL branch + deep log-Mel branch |
| **IADF+ interactive fusion** | 0.236595% | **1.72%** | **0.2312** | Best 2021 LA generalization |
| IADF+AOCloss fixed | **0.156982%** | 1.75% | 0.2338 | Internal ablation with auxiliary center loss |

The current public release focuses on **IADF+ interactive fusion**, which gives the best ASVspoof2021 LA evaluation result in this experiment set.

## Method Overview

```text
raw waveform
   |-- XLS-R/Wav2Vec2 SSL branch
   |       `-- x_ssl: [B, T_ssl, 128]
   |
   |-- deep log-Mel acoustic branch
   |       `-- x_mel: [B, T_ssl, 128]
   |
   |-- CrossFeatureFusion
   |       |-- SSL -> log-Mel cross-attention
   |       |-- log-Mel -> SSL cross-attention
   |       |-- [ssl, mel, ssl * mel, |ssl - mel|]
   |       `-- frame/channel-wise gated fusion
   |
   |-- AASIST-style spectro-temporal graph attention backend
   `-- bonafide / spoof logits
```

### SSL/XLS-R Branch

The SSL branch extracts frame-level representations from raw waveform:

```text
x_ssl_feat: [B, T_ssl, 1024]
x_ssl:      [B, T_ssl, 128]
```

This branch captures high-level speech cues, including phonetic content, speaker-related information, and SSL-space artifacts introduced by spoofing attacks.

### Deep log-Mel Acoustic Branch

The acoustic branch converts waveform into log-Mel features and refines them with:

- linear projection;
- LayerNorm + SELU;
- multi-scale depthwise temporal convolution;
- BiLSTM;
- channel gate;
- temporal interpolation to match the SSL sequence length.

The output is aligned with the SSL branch:

```text
x_mel: [B, T_ssl, 128]
```

This branch emphasizes local spectral texture, energy patterns, compression traces, and vocoder-related acoustic artifacts.

### Interactive Feature Fusion

The fusion module first performs bidirectional cross-attention:

```text
SSL query log-Mel
log-Mel query SSL
```

It then builds explicit interaction features:

```text
[ssl_feat, mel_feat, ssl_feat * mel_feat, abs(ssl_feat - mel_feat)]
```

The product term models feature co-activation, while the absolute-difference term models representation mismatch between the high-level SSL branch and the low-level acoustic branch. A frame/channel-wise gate then produces the fused sequence:

```text
fused = ssl_feat + gate * mixed + (1.0 - gate) * mel_feat
```

### AASIST-style Graph Backend

The fused sequence is converted into a spectro-temporal representation and processed by an AASIST-style backend:

```text
fused_seq: [B, T_ssl, 128]
      |
transpose + unsqueeze
      |
[B, 1, 128, T_ssl]
```

The backend uses CNN residual encoding, spectral graph attention, temporal graph attention, heterogeneous graph attention, and final pooling/classification.

## Repository Structure

```text
.
|-- model.py                  # SSL + deep log-Mel fusion + AASIST-style backend
|-- main_SSL_LA.py            # training and score generation entry
|-- data_utils_SSL.py         # ASVspoof data loading and RawBoost preprocessing
|-- RawBoost.py               # RawBoost data augmentation
|-- eval_metric_LA.py         # LA EER / t-DCF utilities
|-- eval_metrics_DF.py        # DF EER utilities
|-- evaluate_2021_LA.py       # ASVspoof2021 LA evaluation helper
|-- evaluate_2021_DF.py       # ASVspoof2021 DF evaluation helper
|-- core_scripts/
|   `-- startup_config.py
`-- database/
    |-- ASVspoof_LA_cm_protocols/
    `-- ASVspoof_DF_cm_protocols/
```

Large assets are kept outside this repository: ASVspoof audio data, official evaluation keys, XLS-R/Wav2Vec2 weights, trained checkpoints, score files, and training logs.

## Installation

```bash
git clone https://github.com/paradox-ys/aasist-interactive-ssl-logmel-fusion.git
cd aasist-interactive-ssl-logmel-fusion
pip install -r requirements.txt
```

Prepare a local XLS-R/Wav2Vec2 model directory or use a HuggingFace model name. In our experiments, the SSL model path was:

```text
/root/autodl-tmp/xlsr_300m_hf
```

Set your own path with `--ssl_model_path`.

## Data Preparation

Recommended data layout:

```text
/path/to/data/LA/
|-- ASVspoof2019_LA_train/flac/*.flac
|-- ASVspoof2019_LA_dev/flac/*.flac
`-- ASVspoof2019_LA_eval/flac/*.flac

/path/to/data/
|-- ASVspoof2021_LA_eval/flac/*.flac
`-- ASVspoof2021_DF_eval/flac/*.flac
```

Protocol files are stored under `database/`. Use the official ASVspoof keys for final metric computation.

## Training

Train on ASVspoof2019 LA:

```bash
python main_SSL_LA.py \
  --database_path /root/autodl-tmp/data/LA/ \
  --protocols_path database/ \
  --ssl_model_path /root/autodl-tmp/xlsr_300m_hf \
  --batch_size 8 \
  --num_epochs 50 \
  --lr 1e-6 \
  --comment interactive_fusion
```

Checkpoints are saved under:

```text
models/model_LA_weighted_CCE_.../
```

## Evaluation

### ASVspoof2021 LA

Generate scores:

```bash
python main_SSL_LA.py \
  --database_path /root/autodl-tmp/data/ \
  --protocols_path database/ \
  --track LA \
  --eval \
  --model_path models/model_LA_weighted_CCE_50_8_1e-06_interactive_fusion/best_model.pth \
  --eval_output scores_2021_LA.txt
```

Compute metrics:

```bash
python evaluate_2021_LA.py scores_2021_LA.txt ./LA-keys-stage-1/keys eval
```

### ASVspoof2021 DF

Generate scores:

```bash
python main_SSL_LA.py \
  --database_path /root/autodl-tmp/data/ \
  --protocols_path database/ \
  --track DF \
  --eval \
  --model_path models/model_LA_weighted_CCE_50_8_1e-06_interactive_fusion/best_model.pth \
  --eval_output scores_2021_DF.txt
```

Compute EER:

```bash
python evaluate_2021_DF.py scores_2021_DF.txt ./DF-keys-stage-1/keys eval
```

## Implementation Notes

- Label convention follows common ASVspoof / SSL anti-spoofing settings: `bonafide = 1`, `spoof = 0`.
- The checkpoint paths in this README are examples. Replace them with your own trained model paths.
- This code is intended for research and reproducibility experiments on audio deepfake detection.

## References and Acknowledgements

This project builds on ideas and code structure from AASIST, SSL anti-spoofing, RawBoost, and ASVspoof evaluation tooling. Please cite the original AASIST, SSL anti-spoofing, RawBoost, and ASVspoof papers when using this repository in research.

## License

This repository is released under the MIT License.
