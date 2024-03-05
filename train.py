import torch.optim as optim
import torch 

from model import BERT
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

labels = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1]  # 2 classes
max_length = 5  # max length is 5

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



# loss function
criterion = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 300  # define epoch

for epoch in range(num_epochs):
    # train model
    model.train()
    for batch_idx, (data, targets,mask) in enumerate(data_loader):
        data = data.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        scores = model(data, mask=mask)  
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print tqmd
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}')
