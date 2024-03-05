import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)    #2,7,8,32
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)    #2,7,8,32
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just a way to do batch matrix multiplication
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])   #2,8,7,7    [batch_size, num_heads, seq_len_q, seq_len_k]
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out) #2,5,256
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out  #2,5,256

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a long enough P
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Make embeddings relatively larger
        x = x * math.sqrt(self.pe.size(-1))
        # Add constant to embedding
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len]
        pe = pe.repeat(x.size(0), 1, 1)
        return pe



class BERT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(BERT, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = PositionalEncoding(embed_size, max_length)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(256, 2)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        # In the Encoder the query, key, value are all the same, it's in the decoder this will change.
        # This might seem a bit odd in this case, but it's done to keep the interface consistent.
        for layer in self.layers:
            out = layer(out, out, out, mask)
        out = out[:, 0, :]
        scores = self.fc_out(out)
        return scores  #2,5,256

def main():
    # Example of using BERT model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_layers = 2
    vocab_size = 10000  # This should be adjusted to your dataset.
    embed_size = 256
    heads = 8
    forward_expansion = 4
    dropout = 0.1
    max_length = 100  # Max length of a sentence

    model = BERT(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_layers=num_layers,
        heads=heads,
        device=device,
        forward_expansion=forward_expansion,
        dropout=dropout,
        max_length=max_length,
    ).to(device)

    # Dummy input for testing
    x = torch.tensor([[1, 2, 3, 4, 5, 0, 0],[1, 2, 3, 4, 5, 0, 0]]).to(device)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0],[1, 1, 1, 1, 1, 0, 0]]).to(device)

    output = model(x, mask=mask)
    print(output.shape)  # (batch_size, seq_length, embed_size)


import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


vocab = {'[PAD]': 0, '[UNK]': 1, 'hello': 2, 'world': 3, 'good': 4, 'morning': 5, 'evening': 6, 'day': 7, 'night': 8, 'bye': 9}
vocab_size = len(vocab)
# 文本序列和它们的标签
texts = [
    "hello world",
    "good morning",
    "good evening",
    "hello day",
    "good night",
    "bye world",
    "hello night",
    "good day",
    "bye night",
    "hello evening"
]

labels = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1]  # 假设0和1分别代表两个类别
max_length = 5  # 假设最大序列长度是5

def text_to_index(texts, vocab, max_length):
    indices = []
    attention_masks = []

    for text in texts:
        tokens = text.split()
        seq = [vocab.get(token, vocab['[UNK]']) for token in tokens][:max_length]
        padding = [vocab['[PAD]']] * (max_length - len(seq))

        attention_mask = [1] * len(seq) + [0] * len(padding)
        
        seq += padding
        indices.append(seq)
        attention_masks.append(attention_mask)
    
    return torch.tensor(indices), torch.tensor(attention_masks)

x, mask = text_to_index(texts, vocab, max_length)
y = torch.tensor(labels)

from torch.utils.data import DataLoader, TensorDataset
batch_size = 2
# 创建TensorDataset和DataLoader
dataset = TensorDataset(x, y, mask)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Hyperparameters
num_layers = 2
vocab_size = 10000  # This should be adjusted to your dataset.
embed_size = 256
heads = 8
forward_expansion = 4
dropout = 0.1
max_length = 100  # Max length of a sentence

model = BERT(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_layers=num_layers,
    heads=heads,
    device=device,
    forward_expansion=forward_expansion,
    dropout=dropout,
    max_length=max_length,
).to(device)



# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 300  # 定义训练的轮数

for epoch in range(num_epochs):
    # 将模型设置为训练模式
    model.train()
    
    # 假设 'data_loader' 是你的数据加载器
    for batch_idx, (data, targets,mask) in enumerate(data_loader):
        data = data.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        # 前向传播
        scores = model(data, mask=mask)  # 假设你已经有了一个适用于你任务的mask 2,1
        loss = criterion(scores, targets)   #1,1

        # 后向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练进度
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}')

