import random
from typing import Union

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import Wav2Vec2Model


___author__ = "Hemlata Tak (modified: fairseq -> transformers)"
__email__ = "tak@eurecom.fr"

############################
## FOR fine-tuned SSL MODEL (HuggingFace version)
############################

class SSLModel(nn.Module):
    def __init__(self, device, model_path='/root/autodl-tmp/xlsr_300m_hf'):
        super(SSLModel, self).__init__()

        self.model = Wav2Vec2Model.from_pretrained(model_path)
        self.device = device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        if next(self.model.parameters()).device != input_data.device            or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        emb = self.model(input_tmp).last_hidden_state
        return emb


class LogMelBranch(nn.Module):
    def __init__(self, sr=16000, n_fft=512, win_length=400, hop_length=160, n_mels=128, hidden_dim=128):
        super().__init__()
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr / 2)
        self.register_buffer('mel_basis', torch.tensor(mel_basis, dtype=torch.float32))
        self.register_buffer('window', torch.hann_window(win_length), persistent=False)
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.input_proj = nn.Sequential(
            nn.Linear(n_mels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SELU(inplace=True),
        )
        # Multi-scale temporal modeling before sequence encoding.
        self.dw_conv_3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.dw_conv_7 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim)
        self.pw_conv = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.ms_norm = nn.LayerNorm(hidden_dim)

        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SELU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SELU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, wav, target_len):
        if wav.ndim == 3:
            wav = wav[:, :, 0]
        window = self.window.to(device=wav.device, dtype=wav.dtype)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True,
            pad_mode='reflect',
        )
        power = spec.abs().pow(2.0)
        mel_basis = self.mel_basis.to(device=power.device, dtype=power.dtype)
        mel = torch.einsum('mf,bft->bmt', mel_basis, power)
        log_mel = torch.log(mel + 1e-6).transpose(1, 2)

        feats = self.input_proj(log_mel)
        feats_c = feats.transpose(1, 2)
        ms_feat = torch.cat([
            self.dw_conv_3(feats_c),
            self.dw_conv_7(feats_c),
        ], dim=1)
        ms_feat = self.pw_conv(ms_feat).transpose(1, 2)
        feats = self.ms_norm(feats + ms_feat)

        feats, _ = self.encoder(feats)
        gate = self.channel_gate(feats.mean(dim=1)).unsqueeze(1)
        feats = feats * gate
        feats = self.out_proj(feats)
        if feats.size(1) != target_len:
            feats = F.interpolate(feats.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
        return feats


class FeedForwardBlock(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TemporalRefineBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw_conv_3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dw_conv_5 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.mix = nn.Conv1d(dim * 2, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, T, C]
        xc = x.transpose(1, 2)
        refined = torch.cat([self.dw_conv_3(xc), self.dw_conv_5(xc)], dim=1)
        refined = self.mix(refined).transpose(1, 2)
        return self.norm(x + self.act(refined))


class CrossFeatureFusion(nn.Module):
    def __init__(self, dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.ssl_to_mel = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mel_to_ssl = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        self.ssl_norm1 = nn.LayerNorm(dim)
        self.ssl_norm2 = nn.LayerNorm(dim)
        self.mel_norm1 = nn.LayerNorm(dim)
        self.mel_norm2 = nn.LayerNorm(dim)

        self.ssl_ffn = FeedForwardBlock(dim=dim, expansion=4, dropout=dropout)
        self.mel_ffn = FeedForwardBlock(dim=dim, expansion=4, dropout=dropout)

        # Interactive attention fusion: concatenate aligned features, multiplicative interaction,
        # and discrepancy cues, then adaptively gate the fused representation.
        self.mix_proj = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.Sigmoid(),
        )
        self.channel_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid(),
        )
        self.temporal_refine = TemporalRefineBlock(dim)
        self.out_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ssl_feat, mel_feat):
        ssl_ctx, _ = self.ssl_to_mel(ssl_feat, mel_feat, mel_feat)
        mel_ctx, _ = self.mel_to_ssl(mel_feat, ssl_feat, ssl_feat)

        ssl_feat = self.ssl_norm1(ssl_feat + self.dropout(ssl_ctx))
        ssl_feat = self.ssl_norm2(ssl_feat + self.ssl_ffn(ssl_feat))
        mel_feat = self.mel_norm1(mel_feat + self.dropout(mel_ctx))
        mel_feat = self.mel_norm2(mel_feat + self.mel_ffn(mel_feat))

        fusion_input = torch.cat([
            ssl_feat,
            mel_feat,
            ssl_feat * mel_feat,
            torch.abs(ssl_feat - mel_feat),
        ], dim=-1)
        mixed = self.mix_proj(fusion_input)
        gate = self.gate_proj(fusion_input)
        fused = ssl_feat + gate * mixed + (1.0 - gate) * mel_feat

        channel_weight = self.channel_gate(fused.mean(dim=1)).unsqueeze(1)
        fused = fused * channel_weight + ssl_feat
        fused = self.temporal_refine(fused)
        return self.out_norm(fused)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)

        self.input_drop = nn.Dropout(p=0.2)

        self.act = nn.SELU(inplace=True)

        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map(self, x):
        att_map = self._pairwise_mul_nodes(x)
        att_map = self.att_proj(att_map)
        att_map = torch.tanh(att_map)
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)

        self.input_drop = nn.Dropout(p=0.2)

        self.act = nn.SELU(inplace=True)

        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        x = self.input_drop(x)
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        x1 = x[:, :num_type1, :]
        x2 = x[:, num_type1:, :]
        return x1, x2, master

    def _update_master(self, x, master):
        att_map = self._derive_att_map_master(x, master)
        return self._project_master(x, master, att_map)

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map(self, x, num_type1, num_type2):
        att_map = self._pairwise_mul_nodes(x)
        att_map = self.att_proj(att_map)
        att_map = torch.tanh(att_map)
        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(att_map[:, num_type1:, :num_type1, :], self.att_weight12)
        att_map = att_board / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _derive_att_map_master(self, x, master):
        master = master.expand(-1, x.size(1), -1)
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x, master, att_map):
        x1 = self.proj_with_attM(torch.matmul(att_map.transpose(1, 2).squeeze(-1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k, in_dim, p):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)
        return new_h

    def top_k_graph(self, scores, h, k):
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        h = torch.gather(h, 1, idx)
        return h


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0], out_channels=nb_filts[1],
                               kernel_size=(2, 3), padding=(1, 1), stride=1)
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1], out_channels=nb_filts[1],
                               kernel_size=(2, 3), padding=(0, 1), stride=1)
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0], out_channels=nb_filts[1],
                                             padding=(0, 1), kernel_size=(1, 3), stride=1)
        else:
            self.downsample = False

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity
        return out


class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.device = device
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        self.ssl_model = SSLModel(
            self.device,
            model_path=getattr(args, 'ssl_model_path', '/root/autodl-tmp/xlsr_300m_hf'),
        )
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.mel_branch = LogMelBranch(
            sr=getattr(args, 'sample_rate', 16000),
            n_fft=getattr(args, 'mel_n_fft', 512),
            win_length=getattr(args, 'mel_win_length', 400),
            hop_length=getattr(args, 'mel_hop_length', 160),
            n_mels=getattr(args, 'mel_bins', 128),
            hidden_dim=128,
        )
        self.fusion = CrossFeatureFusion(dim=128, num_heads=getattr(args, 'fusion_heads', 4), dropout=0.1)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1,1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1,1)),
        )
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):
        raw_wav = x.squeeze(-1) if x.ndim == 3 else x
        x_ssl_feat = self.ssl_model.extract_feat(raw_wav)
        x_ssl = self.LL(x_ssl_feat)
        x_mel = self.mel_branch(raw_wav, target_len=x_ssl.size(1))
        x = self.fusion(x_ssl, x_mel)
        x = x.transpose(1, 2)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)
        w = self.attention(x)
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
        e_T = m1.transpose(1, 2)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug
        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)
        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        return output
