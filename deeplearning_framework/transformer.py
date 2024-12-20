import math
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Defined in :numref:`sec_attention-scoring-functions`"""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_additive-attention`"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

# @save
class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)  # Dropout层
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))  # 初始化位置编码张量
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)  # 偶数位置使用sin函数
        self.P[:, :, 1::2] = torch.cos(X)  # 奇数位置使用cos函数

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)  # 添加位置编码
        return self.dropout(X)  # 应用Dropout


# 残差连接和层规范化
# 加法和规范化（add&norm）组件
# ln = nn.LayerNorm(3) # 3表示最后一个维度的大小是3
# bn = nn.BatchNorm1d(3) # 3表示输入数据有3个通道。
# X = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.float32)
# 在训练模式下计算X的均值和方差
# print('layer norm:', ln(X), '\nbatch norm:', bn(X))
"""
Layer Normalization对每一行进行归一化处理
layer norm: tensor([[-1.2247,  0.0000,  1.2247],
        [-1.2247,  0.0000,  1.2247]], grad_fn=<NativeLayerNormBackward>)
Batch Normalization对每一列（通道）进行归一化处理
batch norm: tensor([[-1.0000, -1.0000, -1.0000],
        [ 1.0000,  1.0000,  1.0000]], grad_fn=<NativeBatchNormBackward>)
"""


# 残差连接和层规范化来实现AddNorm类。
# 暂退法也被作为正则化方法使用。
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.ln = nn.LayerNorm(normalized_shape)  # LayerNorm层

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)  # 残差连接后进行层规范化


# 残差连接要求两个输入的形状相同，以便加法操作后输出张量的形状相同。
# add_norm = AddNorm([3, 4], 0.5)
# add_norm.eval()
# print(add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape)
# torch.Size([2, 3, 4])


# 多头注意力
# @save
class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads  # 注意力头数
        self.attention = DotProductAttention(dropout)  # 点积注意力
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)  # 线性变换层，用于生成查询向量
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)  # 线性变换层，用于生成键向量
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)  # 线性变换层，用于生成值向量
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)  # 线性变换层，用于生成输出向量

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)  # 变换查询向量
        keys = transpose_qkv(self.W_k(keys), self.num_heads)  # 变换键向量
        values = transpose_qkv(self.W_v(values), self.num_heads)  # 变换值向量

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


# @save
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


# @save
def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class PositionWiseFFN(nn.Module):  # @save
    """基于位置的前馈网络"""

    # 基于位置的前馈网络对序列中的所有位置的表示进行变换时使用的是同一个多层感知机（MLP）
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)  # 第一个全连接层
        self.relu = nn.ReLU()  # ReLU激活函数
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)  # 第二个全连接层

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))  # 前向传播


# 编码器
# EncoderBlock类包含两个子层：多头自注意力和基于位置的前馈网络，
# 这两个子层都使用了残差连接和紧随的层规范化。
# @save
class EncoderBlock(nn.Module):
    """Transformer编码器块"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens,
            num_heads, dropout, use_bias)  # 多头注意力层
        self.addnorm1 = AddNorm(norm_shape, dropout)  # 残差连接和层规范化
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)  # 基于位置的前馈网络
        self.addnorm2 = AddNorm(norm_shape, dropout)  # 残差连接和层规范化

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))  # 多头注意力后进行残差连接和层规范化
        return self.addnorm2(Y, self.ffn(Y))  # 前馈网络后进行残差连接和层规范化


# Transformer编码器中的任何层都不会改变其输入的形状
# X = torch.ones((2, 100, 24))
# valid_lens = torch.tensor([3, 2])
# encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
# encoder_blk.eval()
# encoder_blk(X, valid_lens).shape
# torch.Size([2, 100, 24])

# 实现的Transformer编码器的代码中，堆叠了num_layers个EncoderBlock类的实例
# @save
class TransformerEncoder(d2l.Encoder):
    """Transformer编码器"""

    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens  # 隐藏单元数
        self.embedding = nn.Embedding(vocab_size, num_hiddens)  # 嵌入层
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)  # 位置编码层
        self.blks = nn.Sequential()  # 编码器块的容器
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))  # 添加编码器块

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))  # 嵌入层和位置编码
        self.attention_weights = [None] * len(self.blks)  # 存储注意力权重
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)  # 通过每个编码器块
            self.attention_weights[i] = blk.attention.attention.attention_weights  # 存储每个块的注意力权重
        return X  # 返回编码结果

encoder = TransformerEncoder(
    200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 1, 0.5)
print(encoder)
"""
TransformerEncoder(
  (embedding): Embedding(200, 24)
  (pos_encoding): PositionalEncoding(
    (dropout): Dropout(p=0.5, inplace=False)
  )
  (blks): Sequential(
    (block0): EncoderBlock(
      (attention): MultiHeadAttention(
        (attention): DotProductAttention(
          (dropout): Dropout(p=0.5, inplace=False)
        )
        (W_q): Linear(in_features=24, out_features=24, bias=False)
        (W_k): Linear(in_features=24, out_features=24, bias=False)
        (W_v): Linear(in_features=24, out_features=24, bias=False)
        (W_o): Linear(in_features=24, out_features=24, bias=False)
      )
      (addnorm1): AddNorm(
        (dropout): Dropout(p=0.5, inplace=False)
        (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)
      )
      (ffn): PositionWiseFFN(
        (dense1): Linear(in_features=24, out_features=48, bias=True)
        (relu): ReLU()
        (dense2): Linear(in_features=48, out_features=24, bias=True)
      )
      (addnorm2): AddNorm(
        (dropout): Dropout(p=0.5, inplace=False)
        (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)
      )
    )
  )
)
"""

# 解码器
"""
Transformer解码器也是由多个相同的层组成。
在DecoderBlock类中实现的每个层包含了三个子层：解码器自注意力、“编码器-解码器”注意力和基于位置的前馈网络。
这些子层也都被残差连接和紧随的层规范化围绕。
"""


class DecoderBlock(nn.Module):
    """解码器中第i个块"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i  # 当前块的索引
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)  # 多头注意力层1
        self.addnorm1 = AddNorm(norm_shape, dropout)  # 残差连接和层规范化1
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)  # 多头注意力层2
        self.addnorm2 = AddNorm(norm_shape, dropout)  # 残差连接和层规范化2
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)  # 基于位置的前馈网络
        self.addnorm3 = AddNorm(norm_shape, dropout)  # 残差连接和层规范化3

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]  # 编码器的输出和有效长度
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X  # 当前没有存储的值时，使用输入X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)  # 连接存储的值和输入X
        state[2][self.i] = key_values  # 更新存储的值
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)  # 训练阶段生成有效长度
        else:
            dec_valid_lens = None  # 预测阶段有效长度为None

        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)  # 自注意力
        Y = self.addnorm1(X, X2)  # 残差连接和层规范化1
        # 编码器－解码器注意力。
        # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)  # 编码器－解码器注意力
        Z = self.addnorm2(Y, Y2)  # 残差连接和层规范化2
        return self.addnorm3(Z, self.ffn(Z)), state  # 前馈网络后进行残差连接和层规范化3


# 构建了由num_layers个DecoderBlock实例组成的完整的Transformer解码器
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens  # 隐藏单元数
        self.num_layers = num_layers  # 解码器层数
        self.embedding = nn.Embedding(vocab_size, num_hiddens)  # 嵌入层
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)  # 位置编码层
        self.blks = nn.Sequential()  # 解码器块的容器
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, i))  # 添加解码器块
        self.dense = nn.Linear(num_hiddens, vocab_size)  # 输出层

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))  # 嵌入层和位置编码
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]  # 存储注意力权重
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights  # 返回注意力权重


decoder = TransformerDecoder(
     200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 1, 0.5)
print(decoder)
"""
TransformerDecoder(
  (embedding): Embedding(200, 24)
  (pos_encoding): PositionalEncoding(
    (dropout): Dropout(p=0.5, inplace=False)
  )
  (blks): Sequential(
    (block0): DecoderBlock(
      (attention1): MultiHeadAttention(
        (attention): DotProductAttention(
          (dropout): Dropout(p=0.5, inplace=False)
        )
        (W_q): Linear(in_features=24, out_features=24, bias=False)
        (W_k): Linear(in_features=24, out_features=24, bias=False)
        (W_v): Linear(in_features=24, out_features=24, bias=False)
        (W_o): Linear(in_features=24, out_features=24, bias=False)
      )
      (addnorm1): AddNorm(
        (dropout): Dropout(p=0.5, inplace=False)
        (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)
      )
      (attention2): MultiHeadAttention(
        (attention): DotProductAttention(
          (dropout): Dropout(p=0.5, inplace=False)
        )
        (W_q): Linear(in_features=24, out_features=24, bias=False)
        (W_k): Linear(in_features=24, out_features=24, bias=False)
        (W_v): Linear(in_features=24, out_features=24, bias=False)
        (W_o): Linear(in_features=24, out_features=24, bias=False)
      )
      (addnorm2): AddNorm(
        (dropout): Dropout(p=0.5, inplace=False)
        (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)
      )
      (ffn): PositionWiseFFN(
        (dense1): Linear(in_features=24, out_features=48, bias=True)
        (relu): ReLU()
      (addnorm2): AddNorm(
        (dropout): Dropout(p=0.5, inplace=False)
        (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)
      )
      (ffn): PositionWiseFFN(
        (dense1): Linear(in_features=24, out_features=48, bias=True)
      (addnorm2): AddNorm(
        (dropout): Dropout(p=0.5, inplace=False)
        (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)
      )
      (addnorm2): AddNorm(
        (dropout): Dropout(p=0.5, inplace=False)
        (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)
      (addnorm2): AddNorm(
        (dropout): Dropout(p=0.5, inplace=False)
        (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)
      (addnorm2): AddNorm(
      (addnorm2): AddNorm(
        (dropout): Dropout(p=0.5, inplace=False)
      (addnorm2): AddNorm(
        (dropout): Dropout(p=0.5, inplace=False)
        (dropout): Dropout(p=0.5, inplace=False)
        (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)
        (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)
      )
      )
      (ffn): PositionWiseFFN(
      (ffn): PositionWiseFFN(
        (dense1): Linear(in_features=24, out_features=48, bias=True)
        (dense1): Linear(in_features=24, out_features=48, bias=True)
        (relu): ReLU()
        (dense2): Linear(in_features=48, out_features=24, bias=True)
      )
      (addnorm3): AddNorm(
        (dropout): Dropout(p=0.5, inplace=False)
        (ln): LayerNorm((100, 24), eps=1e-05, elementwise_affine=True)
      )
    )
  )
  (dense): Linear(in_features=24, out_features=200, bias=True)
)
"""

# 训练
# 英语－法语”机器翻译数据集上训练Transformer模型
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10  # 超参数
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()  # 学习率、训练轮数和设备
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4  # 前馈网络和注意力头的参数
key_size, query_size, value_size = 32, 32, 32  # 注意力机制的参数
norm_shape = [32]  # 规范化形状

# train_iter: 训练数据迭代器，用于生成训练批次数据。
# src_vocab: 源语言（英语）的词汇表对象。
# tgt_vocab: 目标语言（法语）的词汇表对象。
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)  # 加载数据

"""
key_size, query_size, value_size: 注意力机制中键、查询和值的维度。
num_hiddens: 隐藏单元数，表示词嵌入的维度。
norm_shape: 层规范化的形状。
ffn_num_input, ffn_num_hiddens: 前馈神经网络的输入和隐藏层的大小。
num_heads: 多头注意力机制的头数。
num_layers: 解码器层数，即解码器块的数量。
dropout: Dropout层的丢弃概率。
"""
encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)  # 初始化编码器
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)  # 初始化解码器
net = d2l.EncoderDecoder(encoder, decoder)  # 初始化编码解码器模型
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)  # 训练模型
plt.show()  # 显示损失曲线
# loss 0.029, 9302.9 tokens/sec on cuda:0

# Transformer模型将一些英语句子翻译成法语，并且计算它们的BLEU分数。
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']  # 英语句子
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']  # 法语翻译
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)  # 进行翻译
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
"""
go . => va !,  bleu 1.000
i lost . => j'ai perdu .,  bleu 1.000
he's calm . => il est calme .,  bleu 1.000
i'm home . => je suis chez moi .,  bleu 1.000
"""
