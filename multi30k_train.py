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

import pickle
# 是 Python 标准库中的一个模块，
# 主要功能是将 Python 对象（如字典、类实例、列表等）转换为二进制数据（字节流）并保存到文件

# 从tqdm模块导入tqdm，用于创建进度条，可视化任务执行进度
from tqdm import tqdm
# 导入random模块，用于生成随机数（如数据打乱、随机采样等）
import random



# <---------------------数据处理部分------------------->
class Vocab:
    #将文本token与整数索引相互映射
    def __init__(self,tokens,min_freq,specials=('<pad>','<eos>','<sos>','<unk>')):
        self.specials = specials#存储特殊标记
        counter = Counter(tokens)#统计每一个token出现的freq
        self.token_to_idx = {}#文本到索引的字典   实例属性
        #为特殊标记添加索引
        for i, token in enumerate(specials):
            self.token_to_idx[token] = i
        #为常规词汇添加索引
        for token,freq in counter.items():
            if freq >= min_freq:
                self.token_to_idx[token] = len(self.token_to_idx)#自动计算索引
        #从索引到token的字典
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        #存储特殊标记的索引 方便快速访问 全是实例属性
        self.unk_idx = self.token_to_idx['<unk>']
        self.pad_idx = self.token_to_idx['<pad>']
        self.sos_idx = self.token_to_idx['<sos>']
        self.eos_idx = self.token_to_idx['<eos>']
    #返回词汇表大小
    def __len__(self):
        return len(self.token_to_idx)
    #将token列表 转化为索引列表
    def encode(self, tokens):
        return [self.token_to_idx.get(token,self.unk_idx) for token in tokens]
    #get方法  默认获取token的索引 不存在 则赋值为Unk的idx
    # 将索引列表转化为token列表(人类可读的字符串)
    def decode(self, indexes):   #要求输入的是列表
        return  [self.idx_to_token[idx] for idx in indexes if idx in self.idx_to_token]
# 下载multi30k数据集
def download_multi30k():
    base_url = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
    files = {
        'train.de':'train.de.gz',
        'train.en':'train.en.gz',
        'val.de':'val.de.gz',
        'val.en':'val.en.gz',
        'test_2016_flickr.de':'test_2016_flickr.de.gz',
        'test_2016_flickr.en':'test_2016_flickr.en.gz',
    }

    os.makedirs('data',exist_ok=True)
    #遍历需要下载的文件
    for  filename,gzname in files.items():
        #数据存储位置
        filepath = f'data/{filename}'
        if not os.path.exists(filepath):
            # 如果路径不存在则下载数据集
            print('downloading---',filename)
            url = base_url + gzname#拼接完整的下载链接
            try:
                urllib.request.urlretrieve(url,f'data/{gzname}')  #第一个下载链接 第二个是存放位置
                with gzip.open(f'data/{gzname}','rb') as f_in:
                    #打开压缩包 并读取内容进f_in
                    with open(filepath,'wb') as f_out:
                        f_out.write(f_in.read())
                #删除压缩包文件
                os.remove(f'data/{gzname}')
            except Exception as e:
                print(f'failed to download {e}')
                return False
    return True

#简单的文本分词器
def tokenize(text):
    #文本小写  删除首尾空格  token安装分割
    return text.lower().strip().split()
# 加载数据  函数
def load_data(src_file,tgt_file):
    # 加载数据文件 返回分词之后的列表
    with open(src_file,'r',encoding='utf-8') as f:
        src_data = [tokenize(line) for line in f]#文件本身是可迭代内容 不需要加readlines
    with open(tgt_file,'r',encoding='utf-8') as f:
        tgt_data = [tokenize(line) for line in f]
    return src_data, tgt_data#返回的是两个列表

class TranslationDataset:
    # 将分词之后的token转化为tensor张量
    def __init__(self,src_data,tgt_data,src_vocab,tgt_vocab,):
        # 转化为实例属性
        self.src_data = src_data  #存储分词之后的源数据
        self.tgt_data = tgt_data
        self.src_vocab = src_vocab  #源语言词汇表 vocab类
        self.tgt_vocab = tgt_vocab
    def __len__(self):
        return len(self.src_data)#返回数据集样本数量
    def __getitem__(self,idx):   #__getitem__ 是官方规定的接口 不可修改
        #根据一个索引获取一个句子  并将其转化为索引列表 最终转化为tensor张量
        # 处理源语言数据：
        # 1. 用源语言词汇表将token列表编码为索引列表
        # 2. 在开头添加句首标记（sos_idx），结尾添加句尾标记（eos_idx）
        src = [self.src_vocab.sos_idx] + self.src_vocab.encode(self.src_data[idx]) + [self.src_vocab.eos_idx]
        tgt = [self.tgt_vocab.sos_idx] + self.tgt_vocab.encode(self.tgt_data[idx]) + [self.tgt_vocab.eos_idx]
        return torch.tensor(src,dtype=torch.long), torch.tensor(tgt,dtype=torch.long)
    # 将索引列表转换为PyTorch长整型张量（模型输入要求为tensor格式）

# 整理函数  将原序列和目标序列 处理为大小一致的批次张量
def collate_fn(batch,src_pad_idx,tgt_pad_idx):
    src_batch,tgt_batch = zip(*batch)
    # 序列最大长度
    max_src_len = max(len(s) for s in src_batch)
    max_tgt_len = max(len(t) for t in tgt_batch)

    src_padded = []#存储原序列填充之后的新序列
    for s in src_batch:
        #torch.full size必须是元组
        padded_tensor = torch.full((max_src_len-len(s),), src_pad_idx, dtype=torch.long)
        padded = torch.concat([s, padded_tensor], dim=0)
        src_padded.append(padded)
    src_padded = torch.stack(src_padded)#将形状相同的张量沿着新的维度堆叠  形成新的维度的张量

    tgt_padded = []#存储目标序列填充之后的新序列
    for t in tgt_batch:
        padded_tensor = torch.full((max_tgt_len-len(t),), tgt_pad_idx, dtype=torch.long)
        padded = torch.concat([t, padded_tensor], dim=0)
        tgt_padded.append(padded)
    tgt_padded = torch.stack(tgt_padded)
    #返回的是两个 高维度张量
    return src_padded, tgt_padded

# =================transformer 模型部分================

#位置编码类
class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionEncoding, self).__init__()
        #生成位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        #生成分子  即位置编码
        # unsqueeze(1) 增加一个维度，变为(max_len, 1)，便于后续广播计算
        position = torch.arange(0, max_len).unsqueeze(1).float()
        # 生成分母  即衰减分子  10000^(-2i/d_model)
        # torch.arange(0, d_model, 2) 表示2i 取所有的偶数维度
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 填充位置编码矩阵
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  #新增加一个批次维度  变为(batch_size,seq_len,d_model)
        self.register_buffer('pe', pe)#将位置编码矩阵注册为缓冲区 不参与参数更新
    def forward(self, x):
        '''
        参数:
            x: 输入序列特征张量，形状为(batch_size, seq_len, d_model)

        返回:
            torch.Tensor: 添加位置编码后的特征张量，形状与输入x一致
                          (batch_size, seq_len, d_model)
        '''
        #截取与输入序列匹配的位置编码
        x = x + self.pe[:, :x.size(1), :]  # 增加 batch 维度的切片（: 表示取所有批次）
        return x

#多头自注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        # 确保d_model能被num_head整除
        assert d_model % num_heads == 0
        # 注册实例属性
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 定义q,k,v的线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)#最终输出的线性投影层
        #使得部分神经元失活  防止模型过拟合
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        #线性投影
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)
        # 拆分多头  将注意力机制拆分到多个子空间并行计算，每个子空间都可以专注于捕获输入序列中不同类型的关联信息
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        #计算缩放点击注意力
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_dim)
        # scores 的形状是(batch_size, n_heads, q_seq_len, k_seq_len)
        # scores[i][j][m][n]表示 “第 i个样本、第j个注意力头中，第m个查询对第n个键的匹配分数”

        # # -------------重点------------
        # 我们需要让每个查询（第m个查询）对所有键（第0~k_seq_len - 1个键）的分数归一化，这样权重之和为1
        # ，才能体现 “对不同键的关注程度占比”

        # -------------------
        if mask is not None:
            # 将掩码为0的位置分数设为-1e9（softmax后接近0，即注意力为0）
            scores = scores.masked_fill(mask == 0, -1e9)#2e7 是科学计数法，代表 20000000（2000 万）
        #计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # 计算最终的注意力向量
        output = torch.matmul(attn_weights, V)

        # 拼接多头  合并为一个张量
        # contiguous确保内存连续，再view重塑为(batch_size, q_seq_len, d_model)
        attn_output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, n_heads, d_k) -> (batch, seq_len, d_model)

        # 线性投影
        output = self.w_o(attn_output) #形状：(batch_size, q_seq_len, d_model)
        return output

# 前馈网络模块
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        # 将输入从d_model(token的维度)映射到更高维度d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# 编码器模块
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        #第一步 自注意力计算 +残差连接 + 层归一化
        # 使用x同时作为Q、K、V，即"自注意力"
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        #第二部 前馈网络+残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

# 解码器模块
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # 自注意力模块
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 交叉注意力模块
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_output, src_mask, tgt_mask):#forward 很多时候 参数来自函数外部
        # tgt_mask屏蔽未来信息  防止信息泄露
        # 自注意力计算
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 交叉注意力计算
        # query 为x key value为 encoder的输出
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        #前馈神经网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x

# transformer模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size,tgt_vocab_size,d_model=512, n_layer=6,
                 d_ff=2048, num_heads=8, dropout=0.1,max_len=5000,src_pad_idx=0,tgt_pad_idx=0):
        super(Transformer, self).__init__()
        # 注册实例属性
        self.d_model = d_model  #token对应的特征向量的大小
        self.src_pad_idx = src_pad_idx#原序列填充符号的索引
        self.tgt_pad_idx = tgt_pad_idx#目标序列填充符号索引

        #嵌入层 将源语言词汇索引 转化为d_model 维度的向量 一个句子对应的索引向量 变为 (batch_size,seq_len,d_model)
        self.src_embedding = nn.Embedding(src_vocab_size, d_model,padding_idx=self.src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model,padding_idx=self.tgt_pad_idx)
        self.pos_encoding = PositionEncoding(d_model, max_len)
        #位置编码层：为序列添加位置信息（Transformer本身无顺序感知能力）

        #编码器层堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(n_layer)
        ])

        #解码器层堆叠
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(n_layer)
        ])

        #将解码器d_model输出 映射到目标词汇表大小
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        #初始化模型权重
        self._init_weights()

    def _init_weights(self):
        #使用xaiver均匀分布初始化所有的参数
        for p in self.parameters():
            if p.dim() > 1:#只初始化权重矩阵 过滤偏置项
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src):
        #生成原序列掩码  屏蔽填充符号 ”pad“
        # src: (batch, src_len)
        # mask: (batch, 1, 1, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        #生成目标序列掩码  屏蔽填充符号
        batch_size, tgt_len = tgt.shape

        #填充掩码
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)

        # 未来信息掩码：下三角矩阵 shape: (tgt_len,tgt_len) 下三角为true 允许关注 上三角为 false 不允许关注
        tgt_future_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()

        # 合并掩码：同时需满足不是PAD且不是未来位置才允许关注，形状(batch_size, 1, tgt_len, tgt_len)
        tgt_mask = tgt_future_mask & tgt_pad_mask

        return tgt_mask

    def encode(self, src, src_mask):
        #解码器前向传播
        #目标序列嵌入 + 缩放 +位置编码 + dropout
        #这里将src的token的index转为为d_model维度向量  方差为一 如果d_ff 则每一个dim数值更小
        #会造成 position_emmbedding信息过多 所以将d_model维度向量 缩放变大
        x = self.dropout(self.pos_encoding(self.src_embedding(src)*math.sqrt(self.d_model)))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_output,src_mask,tgt_mask):
        #解码器前向传播
        x = self.dropout(self.pos_encoding(self.tgt_embedding(tgt)*math.sqrt(self.d_model)))
        for layer in self.decoder_layers:
            x = layer(x, enc_output,src_mask,tgt_mask)
        return x    # 解码器输出，形状(batch_size, tgt_len, d_model)

    def forward(self, src, tgt):#forward 函数 很多时候 参数来自函数外部 除了x
        #生成原序列和目标序列的索引
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output,src_mask,tgt_mask)
        output = self.fc_out(dec_output)

        return output

# ===================== BLEU 评价指标 ====================

def compute_bleu(reference, hypothesis,max_n = 4):
    #计算bleu分数
    #计算参考句子和假设句子的长度
    reference_len = len(reference)
    hypothesis_len = len(hypothesis)

    if hypothesis_len == 0 :
        return 0.0

    #计算n-grams精度
    #存储不同的N-grams的精确度
    precision = []
    for n in range(1, max_n + 1):
        #统计每一个n-grams出现的次数  返回字典
        #参考文本
        ref_ngrams = Counter(
            [tuple(reference[i:i+n]) for i in range(len(reference)-n+1)]
        )
        # 假设文本
        hyp_ngrams = Counter(
            [tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1)]
        )
        #计算假设和参考的重叠总次数  key 和 value 必须完全一致
        overlap = sum((ref_ngrams & hyp_ngrams).values())

        total = sum(hyp_ngrams.values())
        # 假设翻译中所有n-gram 出现的总次数
        if total == 0:
            precision.append(0.0)
        else:
            precision.append(overlap/total)

    #计算几何平均
    # 若任何一个n - gram的精确度为0，整体BLEU分数为0（惩罚完全不匹配）
    if min(precision) == 0:
        return 0.0
    # 几何平均 = exp( (ln(p1) + ln(p2) + ... + ln(pn)) / n )
    geo_mean = math.exp(sum([math.log(p) for p in precision]) / max_n)

    #简短惩罚  防止假设翻译过短导致的虚高分数
    #如果 假设句子 < 参考句子  加上陈发
    if hypothesis_len < reference_len:
        bp = math.exp(1 - reference_len / hypothesis_len)
    else:
        bp = 1
    return geo_mean * bp

#翻译单个句子的模块
def translate_sentence(model,src,vocab_tgt,device,max_len = 50):
    # max_len: 生成翻译的最大长度，防止无限循环
    model.eval()#模型设置为评估模式
    #创建原序列源码
    src_mask = model.make_src_mask(src)
    # 编码器输出原句子的上下文向量表示
    enc_output = model.encode(src, src_mask)

    #初始化目标句子张量 只包含一个开始符
    tgt = torch.tensor([[vocab_tgt.sos_idx]],device=device)

    for _ in range(max_len):
        tgt_mask = model.make_tgt_mask(tgt)#初始化目标句子掩码
        dec_output = model.decode(tgt, enc_output,src_mask,tgt_mask)
        # dec_output的形状会是[batch_size, current_tgt_len, hidden_dim]
        # hidden_dim = d_model  变为 tgt_vocab_size
        output = model.fc_out(dec_output[:,-1])  #只需要输出目标序列的最后一个token 的tgt_vocab_size维度向量
        next_token = output.argmax(dim=-1,keepdim=True)
        # argmax() 它的作用是返回指定维度上的索引
        # keepdim  它决定了执行argmax()操作的目标维度是否应在结果中被保留，即使其大小等于 1
        # 将新生成的token 拼接到目标tgt上面
        tgt = torch.cat([tgt, next_token],dim=1)

        #将pytorch张量 转化为python 数字
        if next_token.item() == vocab_tgt.eos_idx:
            # 如果模型预测了 < eos > 标记的索引，说明翻译已经完成
            break
        # tgt 的形状是 [1, seq_len]，.squeeze(0) 移除 batch 维度，变为 [seq_len]
        # .tolist() 将张量转换为 Python 列表（[idx_1, idx_2, ...])
        # vocab_tgt.decode() 方法将这个索引列表转换回人类可读的字符串
    return vocab_tgt.decode(tgt.squeeze(0).tolist())

#评价模型在给定数据集上翻译结果的平均bleu分数
def evaluate_bleu(model,dataset,vocab_tgt,vocab_src,device,max_samples = 1000):

    # dataset: 评估数据集（包含源语言句子和对应的参考翻译）
    # max_samples: 最大评估样本数（避免样本过多导致计算缓慢，默认1000）

    # 将模型设置为评估模式
    model.eval()
    #存储每一个样本的bleu分数
    bleu_scores = []
    #实际处理的样本数量
    samples = min(len(dataset),max_samples)
    # 随机抽取数据的索引
    indexes = random.sample(list(range(len(dataset))),samples)

    # 关闭梯度计算（评估时无需反向传播，节省内存和计算资源）
    with torch.no_grad():
        # tqdm 为可迭代对象（如列表、range 等）添加一个进度条，
        # 在循环执行时实时显示完成进度、预计剩余时间等信息，提升代码的交互体
        for index in tqdm(indexes,desc="Evaluating BLEU"):
            src , tgt = dataset[index]
            # 给原始句子 增加batch维度
            src = src.unsqueeze(0).to(device)

            # 翻译
            translated = translate_sentence(model,src,vocab_tgt,device,max_len=50)
            # 为参考句子移除 特殊标记
            # tgt  通常是一个 PyTorch 张量（Tensor）或 NumPy 数组  要先转化为list
            reference = [token for token in vocab_tgt.decode(tgt.tolist())
                          if token not in ['<pad>','<sos>','<eos>']]
            hypothesis = [token for token in translated
                          if token not in ['<pad>','<sos>','<eos>']]
            #计算bleu分数
            bleu = compute_bleu(reference,hypothesis)
            bleu_scores.append(bleu)

    return  sum(bleu_scores)/len(bleu_scores) if bleu_scores else 0.0

# ==========  训练部分 ===========

def train_epoch(model,optimizer,criterion,data_loader,device,clip = 1):
    # 设置为训练模式
    model.train()
    # 初始化本周期的总损失
    epoch_loss = 0

    for src,tgt in tqdm(data_loader,desc="Training"):
        src , tgt = src.to(device), tgt.to(device)
        #取目标序列最后一个token的所有部分
        tgt_input = tgt[:, :-1]   #batch_size,seq_len-1
        #取目标序列除第一个token外的所有部分
        tgt_output = tgt[:, 1:]

        #清空优化器中的梯度
        optimizer.zero_grad()
        output = model(src,tgt_input)
# output 从 (batch_size,seq_len-1,vocab_size) 变为 (batch_size*seq_len-1,vocab_size)
        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1) # (batch_size*seq_len - 1)
        #计算模型输出与目标标签
        loss = criterion(output,tgt_output)
        #反向传播 计算损失对模型参数的梯度
        loss.backward()

        # 梯度裁剪：限制所有参数梯度的L2范数不超过clip，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # 更新模型参数：根据梯度和优化器规则更新参数
        optimizer.step()

        epoch_loss += loss.item()  #将这个张量转换为一个Python 原生的数值类型

    return epoch_loss/ len(data_loader)

def evaluate(model,dataloader,criterion,device):
    #测试模型在指定数据集上的性能 平均损失
    model.eval()
    epoch_loss = 0
    #验证阶段  不需要参数跟新
    with torch.no_grad():
        for src,tgt in tqdm(dataloader,desc="Evaluating"):
            src , tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src,tgt_input)
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output,tgt_output)
            epoch_loss = loss.item()

    return epoch_loss/len(dataloader)


def main():
    # 设置随机种子
    torch.manual_seed(42) #pytorch的种子
    np.random.seed(42) #np的种子
    random.seed(42) # python 原生 random的种子

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device".format(device))
    if torch.cuda.is_available():
        print(f"Using CUDA device :{torch.cuda.get_device_name(0)}") #获取第0块GPU的型号
        print(f"CUDA device :{torch.version.cuda}")

    #下载数据
    print('准备下载数据集')
    if not download_multi30k():
        print("请确保数据集已经下载到 data/目录")
        return

    #加载数据
    print("加载数据---")
    train_src_file = 'data/train.de'
    train_tgt_file = 'data/train.en'
    valid_src_file = 'data/val.de'
    valid_tgt_file = 'data/val.en'

    #分词之后的列表
    train_src , train_tgt = load_data(train_src_file,train_tgt_file)
    valid_src , valid_tgt = load_data(valid_src_file,valid_tgt_file)

    # 构建词汇表
    print('构建词汇表')
    src_tokens = [token for sent in train_src for token in sent]
    tgt_tokens = [token for sent in train_tgt for token in sent]
    vocab_src = Vocab(src_tokens,min_freq=2)
    vocab_tgt = Vocab(tgt_tokens,min_freq=2)
    print(f"源语言词汇表大小：{len(vocab_src)}")
    print(f"目标语言词汇表大小：{len(vocab_tgt)}")

    #创建数据集
    train_dataset = TranslationDataset(train_src,train_tgt,vocab_src,vocab_tgt)
    valid_dataset = TranslationDataset(valid_src,valid_tgt,vocab_src,vocab_tgt)

    batch_size = 64
    # wrapper是包装器的意思
    def collate_wrapper(batch):
        #数据加载的辅助函数
        return collate_fn(batch, vocab_src.pad_idx, vocab_tgt.pad_idx)

    #创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_wrapper,
        shuffle=True,
        num_workers=0
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        collate_fn=collate_wrapper,
        shuffle=False,
        num_workers=0
    )

    #创建模型
    print("初始化模型--")
    model = Transformer(
        src_vocab_size=len(vocab_src),
        tgt_vocab_size=len(vocab_tgt),
        d_model=512,
        d_ff=2048,
        dropout=0.1,
        num_heads=8,
        n_layer=6,
        src_pad_idx=vocab_src.pad_idx,
        tgt_pad_idx=vocab_tgt.pad_idx,
    ).to(device)

    print(f"模型参数量:{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    #优化器和损失函数
    optimizer = optim.Adam(model.parameters(),#需要优化的模型参数（所有可训练权重/偏置）
                           lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab_tgt.pad_idx## 忽略的标签索引（这里是目标语言的填充符号
    )

    #训练
    n_epochs = 10
    best_val_loss = float('inf') #初始化 “最佳验证损失” 为一个无穷大值

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")

        train_loss = train_epoch(model, optimizer, criterion,train_loader, device)
        valid_loss = evaluate(model, valid_loader, criterion, device)

        print(f"训练损失 :{train_loss:.4f} | {valid_loss:.4f}")

        #保存最佳 模型
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')

        #每两个epoch 评估一次bleu
        if (epoch + 1) % 2 == 0:
            bleu_score = evaluate_bleu(model, valid_dataset, vocab_src, vocab_tgt, device)
            print(f"BLEU Score :{bleu_score:.4f}")

        #翻译示例
        # 随机选取验证集的索引
        sample_idx = np.random.randint(0, len(valid_dataset)-1)
        src, tgt = valid_dataset[sample_idx]
        src_sent = ' '.join(vocab_src.decode(src.tolist()))
        tgt_sent = ' '.join(vocab_tgt.decode(tgt.tolist()))

        src_input = src.unsqueeze(0).to(device)
        translated = translate_sentence(model, src_input, vocab_tgt, device)
        translated_sent = ' '.join(translated)

        print(f"\n示例翻译")
        print(f"src : {src_sent}")
        print(f"tgt : {tgt_sent}")
        print(f"translated : {translated_sent}")

    print("\n训练完成！！！！！！")

    #保存 词汇表
    print("\n保存词汇表")
    with open('vocab.pkl', 'wb') as f:
        pickle.dump({
            'vocab_src': vocab_src,
            'vocab_tgt': vocab_tgt,
        }, f)
        # dump()函数的作用是将对象转换为二进制字节流，并写入到指定文件中，完成 “序列化” 过程
    print("词汇表已保存到vocab.pkl")

    #最终BLEU评估
    print("\n加载最佳模型进行最终的评估")

    model.load_state_dict(torch.load('best_model.pt'))
    final_bleu = evaluate_bleu(model, valid_dataset, vocab_src, vocab_tgt, device)
    print(f"Final BLEU Score: {final_bleu:.4f}")

if __name__ == '__main__':
    main()





































