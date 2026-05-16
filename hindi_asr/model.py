import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, length):
        pe_pos = self.pe[:length]
        if length > 1:
            pe_neg = self.pe[1:length].flip(0)
            return torch.cat([pe_neg, pe_pos], dim=0)
        return pe_pos

class MultiHeadSelfAttentionWithRelPE(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.u_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.v_bias = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    def _rel_shift(self, x):
        b, h, t, t2 = x.shape
        zero_pad = torch.zeros(b, h, t, 1, device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(b, h, t2 + 1, t)
        x = x_padded[:, :, 1:].view(b, h, t, t2)
        return x[:, :, :, :t]
    def forward(self, x, pos_enc):
        residual = x
        x = self.norm(x)
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        r = self.pos_proj(pos_enc).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        q_u = q + self.u_bias.unsqueeze(1)
        q_v = q + self.v_bias.unsqueeze(1)
        content_score = torch.matmul(q_u, k.transpose(-2, -1))
        pos_score = torch.matmul(q_v, r.transpose(-2, -1))
        pos_score = self._rel_shift(pos_score)
        score = (content_score + pos_score) * self.scale
        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out + residual

class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        x = self.depthwise_conv(x)
        
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return x + residual

class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion)
        self.linear2 = nn.Linear(d_model * expansion, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return 0.5 * x + residual

class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, kernel_size=31, expansion=4, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, expansion, dropout)
        self.attn = MultiHeadSelfAttentionWithRelPE(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, expansion, dropout)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x, pos_enc):
        x = self.ff1(x)
        x = self.attn(x, pos_enc)
        x = self.conv(x)
        x = self.ff2(x)
        x = self.norm(x)
        return x

class Conv2DSubsampler(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.proj = nn.Linear(d_model * (80 // 4), d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.transpose(2,3)
        x = self.conv(x)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class ConformerCTC(nn.Module):
    def __init__(
        self,
        vocab_size=71,
        d_model=384,
        num_heads=6,
        num_blocks=16,
        kernel_size=31,
        expansion=4,
        dropout=0.15,
        max_len=5000
    ):
        super().__init__()
        self.subsampler = Conv2DSubsampler(d_model, dropout)
        self.pos_enc = RelativePositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, kernel_size, expansion, dropout)
            for _ in range(num_blocks)
        ])
        self.ctc_head = nn.Linear(d_model, vocab_size)
    def forward(self, x, mel_lengths):
        x = self.subsampler(x)
        t = x.shape[1]
        pos_enc = self.pos_enc(t)
        for block in self.blocks:
            x = block(x, pos_enc)
        logits = self.ctc_head(x)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        l1 = ((mel_lengths - 1) // 2) + 1
        input_lengths = ((l1 - 1) // 2) + 1
        return log_probs, input_lengths
