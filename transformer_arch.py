import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.permute(0, 2, 1, 3)
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights
    
    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        output = self.combine_heads(attn_output)
        
        output = self.wo(output)
        
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        cross_attn_output, cross_attn_weights = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, self_attn_weights, cross_attn_weights

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_seq_len, dropout=0.1):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        attn_weights = []
        
        for layer in self.layers:
            x, weights = layer(x, mask)
            attn_weights.append(weights)
            
        return x, attn_weights

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size, max_seq_len, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        self_attn_weights = []
        cross_attn_weights = []
        
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, enc_output, src_mask, tgt_mask)
            self_attn_weights.append(self_attn)
            cross_attn_weights.append(cross_attn)
            
        return x, self_attn_weights, cross_attn_weights

class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 d_model=512, 
                 num_heads=8, 
                 d_ff=2048, 
                 num_layers=6, 
                 max_seq_len=5000, 
                 dropout=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, src_vocab_size, max_seq_len, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, tgt_vocab_size, max_seq_len, dropout)
        
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
    def create_masks(self, src, tgt):
        src_padding_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        tgt_seq_len = tgt.size(1)
        
        future_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len)).bool()
        future_mask = future_mask.to(tgt.device)
        
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        
        tgt_mask = future_mask & tgt_padding_mask
        
        return src_padding_mask, tgt_mask
        
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.create_masks(src, tgt)
        
        enc_output, enc_attn_weights = self.encoder(src, src_mask)
        
        dec_output, dec_self_attn_weights, dec_cross_attn_weights = self.decoder(
            tgt, enc_output, src_mask, tgt_mask
        )
        
        logits = self.output_layer(dec_output)
        
        attn_weights = {
            'encoder_attn': enc_attn_weights,
            'decoder_self_attn': dec_self_attn_weights,
            'decoder_cross_attn': dec_cross_attn_weights
        }
        
        return logits, attn_weights
    
    def generate(self, src, max_len=100, sos_idx=2, eos_idx=3):
        batch_size = src.size(0)
        device = src.device
        
        tgt = torch.ones(batch_size, 1).fill_(sos_idx).long().to(device)
        
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        enc_output, enc_attn_weights = self.encoder(src, src_mask)
        
        all_attn_weights = {
            'encoder_attn': enc_attn_weights,
            'decoder_self_attn': [],
            'decoder_cross_attn': []
        }
        
        for i in range(max_len - 1):
            tgt_mask = self.create_masks(src, tgt)[1]
            
            dec_output, dec_self_attn, dec_cross_attn = self.decoder(
                tgt, enc_output, src_mask, tgt_mask
            )
            
            next_token_logits = self.output_layer(dec_output[:, -1])
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            tgt = torch.cat([tgt, next_token], dim=1)
            
            all_attn_weights['decoder_self_attn'].append([w[:, -1:].detach() for w in dec_self_attn])
            all_attn_weights['decoder_cross_attn'].append([w[:, -1:].detach() for w in dec_cross_attn])
            
            if (next_token == eos_idx).all():
                break
                
        return tgt, all_attn_weights

def test_transformer():
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers)
    
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    logits, attn_weights = model(src, tgt)
    
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: [batch_size, tgt_seq_len, tgt_vocab_size] = [{batch_size}, {tgt_seq_len}, {tgt_vocab_size}]")
    
    generated, gen_attn_weights = model.generate(src, max_len=20)
    print(f"Generated sequence shape: {generated.shape}")
    
    return model, src, tgt, logits, attn_weights, generated, gen_attn_weights

if __name__ == "__main__":
    test_transformer()
