import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import math


class PositionalEncoding(nn.Module):
    def __init__(self, seq_length, d_model):
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        self._reset_parameters()

    def _reset_parameters(self):
        pe = torch.zeros(self.seq_length, self.d_model)

        positions = einops.rearrange(torch.arange(self.seq_length),
                                     'seq_length -> seq_length 1')
        dims = torch.arange(self.d_model)
        dims = torch.pow(10000, (2 / self.d_model) * dims)
        sinusoidal_args = positions / dims

        pe_sin = torch.sin(sinusoidal_args)
        pe_cos = torch.cos(sinusoidal_args)

        pe[:, 0::2] = pe_sin[:, 0::2]
        pe[:, 1::2] = pe_cos[:, 1::2]

        pe = einops.rearrange(pe,
                              'seq_length d_model -> 1 seq_length d_model')
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = int(d_model / self.num_heads)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w_q.weight)
        self.w_q.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.w_k.weight)
        self.w_k.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.w_v.weight)
        self.w_v.bias.data.fill_(0)

        nn.init.xavier_uniform_(self.w_out.weight)
        self.w_out.bias.data.fill_(0)

    def _apply_mask(self,
                    attention_logits,
                    encoder_padding_mask=None,
                    decoder_padding_mask=None,
                    decoder_causal_mask=None):

        if encoder_padding_mask is not None:
            encoder_padding_mask = einops.rearrange(encoder_padding_mask,
                                                    'batch seq_len -> batch 1 1 seq_len')
            attention_logits = attention_logits.masked_fill(encoder_padding_mask == 0, -9e15)

        if decoder_padding_mask is not None:
            decoder_padding_mask = einops.rearrange(decoder_padding_mask,
                                                    'batch seq_len -> batch 1 1 seq_len')
            attention_logits = attention_logits.masked_fill(decoder_padding_mask == 0, -9e15)

        if decoder_causal_mask is not None:
            attention_logits = attention_logits.masked_fill(decoder_causal_mask == 0, -9e15)

        return attention_logits

    def forward(self,
                query,
                value=None,
                encoder_padding_mask=None,
                decoder_padding_mask=None,
                decoder_causal_mask=None):
        q_proj = self.w_q(query)
        if value is None:
            value = query
        k_proj = self.w_k(value)
        v_proj = self.w_v(value)

        q_proj = einops.rearrange(q_proj,
                                  'batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim',
                                  num_heads=self.num_heads,
                                  head_dim=self.d_head
                                  )

        k_proj = einops.rearrange(k_proj,
                                  'batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim',
                                  num_heads=self.num_heads,
                                  head_dim=self.d_head
                                  )
        k_proj = einops.rearrange(k_proj,
                                  'batch num_heads seq_len head_dim -> batch num_heads head_dim seq_len')

        v_proj = einops.rearrange(v_proj,
                                  'batch seq_len (num_heads head_dim) -> batch num_heads seq_len head_dim',
                                  num_heads=self.num_heads,
                                  head_dim=self.d_head
                                  )

        attention_logits = torch.matmul(q_proj, k_proj) / math.sqrt(q_proj.size()[-1])
        attention_logits = self._apply_mask(attention_logits,
                                            encoder_padding_mask,
                                            decoder_padding_mask,
                                            decoder_causal_mask)

        weights = F.softmax(attention_logits, dim=-1)

        output = torch.matmul(weights, v_proj)
        output = einops.rearrange(output,
                                  'batch num_heads seq_len head_dim -> batch seq_len (num_heads head_dim)')
        output = self.w_out(output)

        return output, weights


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(num_heads, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.pff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attended_x, _ = self.multi_head_attention(x, encoder_padding_mask=mask)
        x = x + self.dropout1(attended_x)
        x = self.norm1(x)

        pff_x = self.pff(x)
        x = x + self.dropout2(pff_x)
        x = self.norm2(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderBlock, self).__init__()
        self.multi_head_attention1 = MultiHeadAttention(num_heads, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.multi_head_attention2 = MultiHeadAttention(num_heads, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.pff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,
                encoder_feats,
                decoder_x,
                encoder_padding_mask=None,
                decoder_padding_mask=None,
                decoder_causal_mask=None):
        decoder_attn_o, _ = self.multi_head_attention1(decoder_x,
                                                       decoder_padding_mask=decoder_padding_mask,
                                                       decoder_causal_mask=decoder_causal_mask)

        decoder_x = decoder_x + self.dropout1(decoder_attn_o)
        decoder_x = self.norm1(decoder_x)

        encoder_decoder_x, _ = self.multi_head_attention2(decoder_x,
                                                          encoder_feats,
                                                          encoder_padding_mask)
        x = decoder_x + self.dropout2(encoder_decoder_x)
        x = self.norm2(x)

        pff_x = self.pff(x)
        x = x + self.dropout3(pff_x)
        x = self.norm3(x)

        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, dropout):
        super(Encoder, self).__init__()
        self.encoders = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, 2 * d_model, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for e in self.encoders:
            x = e(x, mask)

        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, seq_len, dropout):
        super(Decoder, self).__init__()
        self.decoders = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, 2 * d_model, dropout) for _ in range(num_layers)])

        self.register_buffer('causal_mask',
                             1 - torch.triu(torch.ones(seq_len, seq_len), diagonal=1),
                             persistent=False
                             )

    def forward(self,
                encoder_feats,
                decoder_x,
                encoder_padding_mask=None,
                decoder_padding_mask=None):
        for d in self.decoders:
            decoder_x = d(encoder_feats,
                          decoder_x,
                          encoder_padding_mask,
                          decoder_padding_mask,
                          self.causal_mask
                          )

        return decoder_x


class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size, seq_length, enable_padding=False):
        super(Embedding, self).__init__()
        if enable_padding:
            self.embeddings = nn.Embedding(vocab_size, d_model, padding_idx=0)
        else:
            self.embeddings = nn.Embedding(vocab_size, d_model)

        self.positional_embeddings = PositionalEncoding(seq_length, d_model)

    def forward(self, x):
        return self.embeddings(x) + self.positional_embeddings(x)
