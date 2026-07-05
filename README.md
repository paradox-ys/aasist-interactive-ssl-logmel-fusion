# Audio Deepfake Detection: AASIST Baseline with Interactive SSL and log-Mel Feature Fusion

> 基于 AASIST 基线的 SSL 与深层 log-Mel 交互融合伪造语音检测

本仓库是一个面向伪造语音检测的研究型实现。需要特别说明的是，**AASIST 原始模型并不是 SSL 模型**；它的核心优势在于使用图注意力机制建模时域与频域伪造伪影。

本项目以 AASIST 作为检测后端基线，在其前端引入 XLS-R/Wav2Vec2 自监督语音表征和深层 log-Mel 声学表征，并设计交互式融合模块，使模型能够显式建模高层语音表征与底层声学纹理之间的一致性和差异性。

因此，本项目的重点不是“简单把 SSL 接到 AASIST”，而是在 AASIST baseline 上进行特征融合优化：

- 保留 AASIST 对时域、频域伪造痕迹的图注意力建模能力；
- 引入 XLS-R/Wav2Vec2 SSL 前端，提取更高层的语音表征；
- 构建深层 log-Mel 分支，补充频谱能量、局部声学纹理和压缩痕迹；
- 通过双向 cross-attention、交互差异建模和门控融合，让 SSL 表征与 log-Mel 表征充分交互；
- 将融合后的时频序列送入 AASIST-style 图注意力后端完成 bonafide/spoof 分类。

## Core Idea

深度伪造语音往往不是所有特征都同时异常。有些样本在高层语义、音素或说话人表征上看起来较自然，但在局部频谱纹理、能量变化、压缩痕迹或声码器细节上存在异常；也有些样本在短时频谱上较平滑，却在 SSL 表征的帧级变化中暴露不自然的动态模式。

因此，本项目将模型拆成两条互补分支：

```text
raw waveform
   |-- SSL/XLS-R branch ----------- high-level speech representation
   |-- deep log-Mel branch -------- acoustic and spectral artifact representation
                 |
        interactive feature fusion
                 |
        AASIST-style graph attention backend
                 |
          bonafide / spoof
```

核心假设是：**伪造语音的判别线索不仅存在于单一特征内部，也存在于高层语音表征与底层声学表征之间的不一致关系中。**

## Method Overview

### 1. SSL/XLS-R Branch

原始 waveform 输入 XLS-R/Wav2Vec2，得到帧级自监督语音表征：

```text
x_ssl_feat: [B, T_ssl, 1024]
```

随后通过线性层投影到 128 维：

```text
x_ssl: [B, T_ssl, 128]
```

这一分支主要提供发音、音素、说话人、语义上下文以及合成伪影相关的高层语音信息。

### 2. Deep log-Mel Acoustic Branch

另一条分支从原始 waveform 计算 log-Mel 特征，再经过：

- Linear projection；
- LayerNorm + SELU；
- 多尺度 depthwise temporal convolution；
- BiLSTM；
- channel gate；
- 时间维插值对齐。

最终得到与 SSL 分支时间长度一致的声学表示：

```text
x_mel: [B, T_ssl, 128]
```

这一分支更关注底层声学线索，例如频谱能量、局部纹理、压缩痕迹、声码器伪影以及相位连续性造成的间接影响。

### 3. Interactive Dual-branch Feature Fusion

融合模块不是简单拼接，而是先进行双向 cross-attention：

```text
SSL query log-Mel
log-Mel query SSL
```

这一步使高层 SSL 表征能够主动关注底层声学细节，同时也让 log-Mel 声学特征对齐更高层的语音上下文。

随后构造显式交互特征：

```text
[ssl_feat, mel_feat, ssl_feat * mel_feat, abs(ssl_feat - mel_feat)]
```

其中：

- `ssl_feat * mel_feat` 用于建模两类特征的一致性和共现关系；
- `abs(ssl_feat - mel_feat)` 用于建模高层语音表征与底层声学表征之间的不一致。

最后通过逐帧、逐通道的 gate 进行自适应融合：

```text
fused = ssl_feat + gate * mixed + (1.0 - gate) * mel_feat
```

这样模型可以在不同时间片和不同通道上动态决定更依赖 SSL、log-Mel，还是二者的交互差异。

### 4. AASIST-style Graph Attention Backend

融合后的序列被转换为类似时频图的输入：

```text
fused_seq: [B, T_ssl, 128]
      |
transpose + unsqueeze
      |
[B, 1, 128, T_ssl]
```

随后进入 AASIST-style 后端：

- CNN residual encoder 提取局部伪影；
- spectral graph 建模频率维伪造线索；
- temporal graph 建模时间维伪造线索；
- heterogeneous graph attention 建模时域与频域线索之间的交互；
- 分类头输出 bonafide/spoof logits。

## Experimental Results

The models were trained on ASVspoof2019 LA train, selected on ASVspoof2019 LA dev, and evaluated on ASVspoof2021 LA eval. Lower values are better.

| Model variant | ASVspoof2019 LA dev EER | ASVspoof2021 LA EER | ASVspoof2021 LA min t-DCF | Note |
|---|---:|---:|---:|---|
| log-Mel fusion | 0.579153% | 2.28% | 0.2388 | SSL branch + deep log-Mel branch |
| **IADF+ interactive fusion** | 0.236595% | **1.72%** | **0.2312** | Best 2021 LA generalization result |
| IADF+AOCloss fixed | **0.156982%** | 1.75% | 0.2338 | Best dev EER, slightly worse on 2021 LA |

In the current cleaned public version, the main released model is **IADF+ interactive fusion**. The AOCloss row is kept as an internal ablation reference: it improves the 2019 LA dev EER, but the IADF+ model gives the best ASVspoof2021 LA evaluation result.

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

本仓库不包含 ASVspoof 音频数据、官方 evaluation keys、XLS-R 权重文件、训练日志或模型 checkpoint。

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

You also need a local XLS-R/Wav2Vec2 model directory or a HuggingFace model name. In our experiments, the local path was:

```text
/root/autodl-tmp/xlsr_300m_hf
```

If your model path is different, set it with `--ssl_model_path`.

## Data Layout

The training script uses ASVspoof2019 LA train/dev by default.

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

Protocol files are stored under `database/`.

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

Checkpoints will be saved under:

```text
models/model_LA_weighted_CCE_.../
```

## ASVspoof2021 LA Evaluation

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

Compute official-style metrics:

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
  --model_path models/model_LA_weighted_CCE_50_8_1e-06_interactive_fusion/best_model.pth \
  --eval_output scores_2021_DF.txt
```

Compute EER:

```bash
python evaluate_2021_DF.py scores_2021_DF.txt ./DF-keys-stage-1/keys eval
```

## Notes

- Label convention follows common ASVspoof / SSL anti-spoofing settings: `bonafide = 1`, `spoof = 0`.
- This repository releases only code and lightweight protocol files. It does not release datasets, official keys, checkpoints, logs, or private experiment artifacts.
- The checkpoint paths in this README are examples and should be replaced with your own trained model paths.
- If you use this repository for research, please also cite AASIST, SSL anti-spoofing, ASVspoof, and related audio deepfake detection work.

## Acknowledgements

This project is modified and extended from SSL anti-spoofing and AASIST-style baselines. We thank the related open-source projects for supporting research on audio deepfake detection.
