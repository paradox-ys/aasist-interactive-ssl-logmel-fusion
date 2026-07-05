# Audio Deepfake Detection: 基于 SSL+AASIST 的深层 log-Mel 交互融合优化

本仓库是一个面向伪造语音检测的研究型 baseline。模型以 **AASIST** 作为后端基线，在其时频图注意力检测框架上加入 **自监督语音表征 SSL/XLS-R** 和 **深层 log-Mel 声学分支**，并通过交互式注意力模块进行融合，用于提升 ASVspoof 场景下的伪造语音检测能力。

简单来说，本项目不是从零实现一个全新的检测器，而是在 AASIST baseline 的基础上做结构优化：

- 保留 AASIST 对时域/频域伪造痕迹的图注意力建模能力；
- 引入 XLS-R/Wav2Vec2 自监督前端，提取高层语音表征；
- 加入深层 log-Mel 分支，补充底层频谱和声学纹理线索；
- 使用双向 cross-attention 和门控融合建模 SSL 与 log-Mel 之间的一致性和差异性；
- 最终将融合后的序列送入 AASIST-style 图注意力后端完成真假分类。

## 方法概述

模型包含四个核心部分。

1. **SSL/XLS-R 分支**

   原始 waveform 输入 XLS-R/Wav2Vec2，得到帧级自监督语音表征，然后通过线性层投影到 128 维。这一路主要提供发音、音素、说话人和生成伪影相关的高层语音信息。

2. **深层 log-Mel 声学分支**

   原始 waveform 先转换为 log-Mel 特征，再经过线性投影、LayerNorm、SELU、多尺度 depthwise temporal convolution、BiLSTM 和 channel gate。该分支更关注频谱能量、局部声学纹理、压缩痕迹和声码器伪影等底层线索。

3. **交互式特征融合模块**

   融合模块不是简单拼接，而是先进行双向 cross-attention：

   - SSL query log-Mel，使高层语音表征关注底层声学细节；
   - log-Mel query SSL，使声学特征对齐高层语音表征。

   随后构造如下交互特征：

   ```text
   [ssl_feat, mel_feat, ssl_feat * mel_feat, abs(ssl_feat - mel_feat)]
   ```

   其中：

   - `ssl_feat * mel_feat` 用于建模两类特征的一致性；
   - `abs(ssl_feat - mel_feat)` 用于显式建模高层语音表征与底层声学表征之间的不一致。

   对于伪造语音，常见情况是语义或发音表征看起来较自然，但局部声学纹理、能量变化或频谱细节存在异常，因此这种差异建模对深度伪造音频检测有意义。

4. **AASIST-style 图注意力后端**

   融合后的序列被转换为类似时频图的输入，经过 CNN residual encoder 后，分别构建频率维图和时间维图，再通过异构图注意力层建模时域伪影、频域伪影以及二者之间的交互关系，最终输出 bonafide/spoof 二分类结果。

## 仓库结构

```text
.
|-- model.py                  # SSL + 深层 log-Mel 融合 + AASIST-style 图后端
|-- main_SSL_LA.py            # 训练与生成 score 文件的主入口
|-- data_utils_SSL.py         # ASVspoof 数据读取与 RawBoost 预处理
|-- RawBoost.py               # RawBoost 数据增强
|-- eval_metric_LA.py         # LA 任务 EER / t-DCF 计算工具
|-- eval_metrics_DF.py        # DF 任务 EER 计算工具
|-- evaluate_2021_LA.py       # ASVspoof2021 LA 官方风格评测脚本
|-- evaluate_2021_DF.py       # ASVspoof2021 DF 官方风格评测脚本
|-- core_scripts/
|   `-- startup_config.py
`-- database/
    |-- ASVspoof_LA_cm_protocols/
    `-- ASVspoof_DF_cm_protocols/
```

本仓库不包含 ASVspoof 音频数据、官方 evaluation keys、XLS-R 权重文件、训练日志或模型 checkpoint。

## 环境依赖

安装依赖：

```bash
pip install -r requirements.txt
```

还需要准备本地 XLS-R/Wav2Vec2 模型目录，或直接使用 HuggingFace 模型名。实验中使用的是本地路径：

```text
/root/autodl-tmp/xlsr_300m_hf
```

如果你的路径不同，可以通过 `--ssl_model_path` 指定。

## 数据目录格式

训练脚本默认使用 ASVspoof2019 LA 的 train/dev 集进行训练和验证。推荐的数据目录格式如下：

```text
/path/to/data/LA/
|-- ASVspoof2019_LA_train/flac/*.flac
|-- ASVspoof2019_LA_dev/flac/*.flac
`-- ASVspoof2019_LA_eval/flac/*.flac    # 可选

/path/to/data/
|-- ASVspoof2021_LA_eval/flac/*.flac
`-- ASVspoof2021_DF_eval/flac/*.flac
```

协议文件位于 `database/` 目录下。

## 训练示例

在 ASVspoof2019 LA 上训练：

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

训练完成后，模型会保存在：

```text
models/model_LA_weighted_CCE_.../
```

## ASVspoof2021 LA 测试

生成 2021 LA score：

```bash
python main_SSL_LA.py \
  --database_path /root/autodl-tmp/data/ \
  --protocols_path database/ \
  --track LA \
  --eval \
  --model_path models/model_LA_weighted_CCE_50_8_1e-06_logmel_fusion/best_model.pth \
  --eval_output scores_2021_LA.txt
```

使用官方 key 计算指标：

```bash
python evaluate_2021_LA.py scores_2021_LA.txt ./LA-keys-stage-1/keys eval
```

## ASVspoof2021 DF 测试

生成 2021 DF score：

```bash
python main_SSL_LA.py \
  --database_path /root/autodl-tmp/data/ \
  --protocols_path database/ \
  --track DF \
  --eval \
  --model_path models/model_LA_weighted_CCE_50_8_1e-06_logmel_fusion/best_model.pth \
  --eval_output scores_2021_DF.txt
```

使用官方 key 计算 EER：

```bash
python evaluate_2021_DF.py scores_2021_DF.txt ./DF-keys-stage-1/keys eval
```

## 说明

- 标签约定沿用 ASVspoof / SSL baseline 常见设置：`bonafide = 1`，`spoof = 0`。
- 本仓库只发布代码和轻量协议文件，不发布数据集、官方 keys、模型权重或实验日志。
- README 中的 checkpoint 路径仅为示例，需要替换为你自己的训练结果。
- 如果使用本仓库进行论文实验，请同时引用 AASIST、SSL anti-spoofing 以及 ASVspoof 相关工作。

## 致谢

本项目基于 SSL anti-spoofing 与 AASIST baseline 体系进行修改和扩展。感谢相关开源工作对深度伪造语音检测研究的推动。
