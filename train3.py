import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from collections import Counter
import urllib.request
import gzip
import os
from tqdm import tqdm
import random


# ==================== 数据处理部分 ====================
class Vocab:
    def __init__(self, tokens, min_freq=2, specials=('<pad>', '<sos>', '<eos>', '<unk>')):
        self.specials = specials
        counter = Counter(tokens)
        self.token_to_idx = {}

        # 添加特殊标记
        for i, token in enumerate(specials):
            self.token_to_idx[token] = i

        # 添加常规词汇
        for token, freq in counter.items():
            if freq >= min_freq:
                self.token_to_idx[token] = len(self.token_to_idx)

        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.unk_idx = self.token_to_idx['<unk>']
        self.pad_idx = self.token_to_idx['<pad>']
        self.sos_idx = self.token_to_idx['<sos>']
        self.eos_idx = self.token_to_idx['<eos>']

    def __len__(self):
        return len(self.token_to_idx)

    def encode(self, tokens):
        return [self.token_to_idx.get(token, self.unk_idx) for token in tokens]

    def decode(self, indices):
        return [self.idx_to_token[idx] for idx in indices if idx in self.idx_to_token]


def download_multi30k():
    """下载 Multi30k 数据集"""
    base_url = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
    files = {
        'train.de': 'train.de.gz',
        'train.en': 'train.en.gz',
        'val.de': 'val.de.gz',
        'val.en': 'val.en.gz',
        'test_2016_flickr.de': 'test_2016_flickr.de.gz',
        'test_2016_flickr.en': 'test_2016_flickr.en.gz'
    }

    os.makedirs('data', exist_ok=True)

    for filename, gzname in files.items():
        filepath = f'data/{filename}'
        if not os.path.exists(filepath):
            print(f"下载 {filename}...")
            url = base_url + gzname
            try:
                urllib.request.urlretrieve(url, f'data/{gzname}')
                with gzip.open(f'data/{gzname}', 'rb') as f_in:
                    with open(filepath, 'wb') as f_out:
                        f_out.write(f_in.read())
                os.remove(f'data/{gzname}')
            except Exception as e:
                print(f"下载失败: {e}")
                print("请手动下载数据集到 data/ 目录")
                return False
    return True


def tokenize(text):
    """简单的分词器"""
    return text.lower().strip().split()


def load_data(src_file, tgt_file):
    """加载数据"""
    with open(src_file, 'r', encoding='utf-8') as f:
        src_data = [tokenize(line) for line in f]
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_data = [tokenize(line) for line in f]
    return src_data, tgt_data


class TranslationDataset:
    def __init__(self, src_data, tgt_data, src_vocab, tgt_vocab):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src = [self.src_vocab.sos_idx] + self.src_vocab.encode(self.src_data[idx]) + [self.src_vocab.eos_idx]
        tgt = [self.tgt_vocab.sos_idx] + self.tgt_vocab.encode(self.tgt_data[idx]) + [self.tgt_vocab.eos_idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_fn(batch, src_pad_idx, tgt_pad_idx):
    """批处理函数"""
    src_batch, tgt_batch = zip(*batch)

    max_src_len = max(len(s) for s in src_batch)
    max_tgt_len = max(len(t) for t in tgt_batch)

    src_padded = []
    for s in src_batch:
        padded = torch.cat([s, torch.full((max_src_len - len(s),), src_pad_idx, dtype=torch.long)])
        src_padded.append(padded)
    src_padded = torch.stack(src_padded)

    tgt_padded = []
    for t in tgt_batch:
        padded = torch.cat([t, torch.full((max_tgt_len - len(t),), tgt_pad_idx, dtype=torch.long)])
        tgt_padded.append(padded)
    tgt_padded = torch.stack(tgt_padded)

    return src_padded, tgt_padded


# ==================== Transformer 模型部分 ====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into multiple heads
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, n_heads, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(attn_output)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self attention with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Self attention with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross attention with residual connection and layer norm
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8,
                 n_layers=6, d_ff=2048, dropout=0.1, max_len=5000, src_pad_idx=0, tgt_pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        # src: (batch, src_len)
        # mask: (batch, 1, 1, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        # tgt: (batch, tgt_len)
        batch_size, tgt_len = tgt.shape

        # Padding mask: (batch, 1, 1, tgt_len)
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)

        # No-peek mask (lower triangular): (tgt_len, tgt_len)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()

        # Combine masks: (batch, 1, tgt_len, tgt_len)
        tgt_mask = tgt_pad_mask & tgt_sub_mask

        return tgt_mask

    def encode(self, src, src_mask):
        x = self.dropout(self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model)))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.dropout(self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        output = self.fc_out(dec_output)
        return output


# ==================== BLEU 评价指标 ====================
def compute_bleu(reference, hypothesis, max_n=4):
    """计算 BLEU 分数"""
    ref_len = len(reference)
    hyp_len = len(hypothesis)

    if hyp_len == 0:
        return 0.0

    # 计算 n-gram 精确度
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter([tuple(reference[i:i + n]) for i in range(len(reference) - n + 1)])
        hyp_ngrams = Counter([tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) - n + 1)])

        overlap = sum((hyp_ngrams & ref_ngrams).values())
        total = sum(hyp_ngrams.values())

        if total == 0:
            precisions.append(0)
        else:
            precisions.append(overlap / total)

    # 计算几何平均
    if min(precisions) == 0:
        return 0.0

    geo_mean = math.exp(sum([math.log(p) for p in precisions]) / max_n)

    # 简短惩罚
    if hyp_len < ref_len:
        bp = math.exp(1 - ref_len / hyp_len)
    else:
        bp = 1.0

    return bp * geo_mean


def evaluate_bleu(model, dataset, vocab_src, vocab_tgt, device, max_samples=1000):
    """评估 BLEU 分数"""
    model.eval()
    bleu_scores = []

    samples = min(len(dataset), max_samples)
    indices = random.sample(range(len(dataset)), samples)

    with torch.no_grad():
        for idx in tqdm(indices, desc="评估 BLEU"):
            src, tgt = dataset[idx]
            src = src.unsqueeze(0).to(device)

            # 翻译
            translated = translate_sentence(model, src, vocab_tgt, device)

            # 移除特殊标记
            reference = [token for token in vocab_tgt.decode(tgt.tolist())
                         if token not in ['<pad>', '<sos>', '<eos>']]
            hypothesis = [token for token in translated
                          if token not in ['<pad>', '<sos>', '<eos>']]

            bleu = compute_bleu(reference, hypothesis)
            bleu_scores.append(bleu)

    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0


def translate_sentence(model, src, vocab_tgt, device, max_len=50):
    """翻译单个句子"""
    model.eval()
    src_mask = model.make_src_mask(src)
    enc_output = model.encode(src, src_mask)

    tgt = torch.tensor([[vocab_tgt.sos_idx]], device=device)

    for _ in range(max_len):
        tgt_mask = model.make_tgt_mask(tgt)
        dec_output = model.decode(tgt, enc_output, src_mask, tgt_mask)
        output = model.fc_out(dec_output[:, -1])
        next_token = output.argmax(dim=-1, keepdim=True)
        tgt = torch.cat([tgt, next_token], dim=1)

        if next_token.item() == vocab_tgt.eos_idx:
            break

    return vocab_tgt.decode(tgt.squeeze(0).tolist())


# ==================== 训练部分 ====================
def train_epoch(model, dataloader, optimizer, criterion, device, clip=1.0):
    model.train()
    epoch_loss = 0

    for src, tgt in tqdm(dataloader, desc="训练"):
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_input)

        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="验证"):
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)

            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


# ==================== 主函数 ====================
def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")

    # 下载数据
    print("准备数据集...")
    if not download_multi30k():
        print("请确保数据集已下载到 data/ 目录")
        return

    # 加载数据
    print("加载数据...")
    train_src, train_tgt = load_data('data/train.de', 'data/train.en')
    val_src, val_tgt = load_data('data/val.de', 'data/val.en')

    # 构建词汇表
    print("构建词汇表...")
    src_tokens = [token for sent in train_src for token in sent]
    tgt_tokens = [token for sent in train_tgt for token in sent]
    vocab_src = Vocab(src_tokens, min_freq=2)
    vocab_tgt = Vocab(tgt_tokens, min_freq=2)
    print(f"源语言词汇表大小: {len(vocab_src)}")
    print(f"目标语言词汇表大小: {len(vocab_tgt)}")

    # 创建数据集和数据加载器
    train_dataset = TranslationDataset(train_src, train_tgt, vocab_src, vocab_tgt)
    val_dataset = TranslationDataset(val_src, val_tgt, vocab_src, vocab_tgt)

    batch_size = 64

    def collate_wrapper(batch):
        return collate_fn(batch, vocab_src.pad_idx, vocab_tgt.pad_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_wrapper, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_wrapper, num_workers=0
    )

    # 创建模型
    print("初始化模型...")
    model = Transformer(
        src_vocab_size=len(vocab_src),
        tgt_vocab_size=len(vocab_tgt),
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=0.1,
        src_pad_idx=vocab_src.pad_idx,
        tgt_pad_idx=vocab_tgt.pad_idx
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_tgt.pad_idx)

    # 训练
    n_epochs = 10
    best_val_loss = float('inf')

    print("开始训练...")
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print("保存最佳模型!")

        # 每2个epoch评估一次BLEU
        if (epoch + 1) % 2 == 0:
            bleu_score = evaluate_bleu(model, val_dataset, vocab_src, vocab_tgt, device, max_samples=500)
            print(f"BLEU 分数: {bleu_score:.4f}")

        # 翻译示例
        sample_idx = random.randint(0, len(val_dataset) - 1)
        src, tgt = val_dataset[sample_idx]
        src_sent = ' '.join(vocab_src.decode(src.tolist()))
        tgt_sent = ' '.join(vocab_tgt.decode(tgt.tolist()))

        src_input = src.unsqueeze(0).to(device)
        translated = translate_sentence(model, src_input, vocab_tgt, device)
        translated_sent = ' '.join(translated)

        print(f"\n示例翻译:")
        print(f"源: {src_sent}")
        print(f"目标: {tgt_sent}")
        print(f"翻译: {translated_sent}")

    print("\n训练完成!")

    # 保存词汇表
    print("\n保存词汇表...")
    import pickle
    with open('vocab.pkl', 'wb') as f:
        pickle.dump({
            'vocab_src': vocab_src,
            'vocab_tgt': vocab_tgt
        }, f)
    print("词汇表已保存到 vocab.pkl")

    # 最终BLEU评估
    print("\n加载最佳模型进行最终评估...")
    model.load_state_dict(torch.load('best_model.pt'))
    final_bleu = evaluate_bleu(model, val_dataset, vocab_src, vocab_tgt, device, max_samples=1000)
    print(f"最终 BLEU 分数: {final_bleu:.4f}")


if __name__ == "__main__":
    main()