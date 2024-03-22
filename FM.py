import torch
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from utils import create_dataset
dataset = create_dataset('criteo', device=device)
data = dataset.train_valid_test_split()
field_dims, (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = data

class model(torch.nn.Module):
    def __init__(self, field_dims, k, embed_dim=4):
        super(model, self).__init__()
        self.n = len(field_dims)*embed_dim
        self.k = k
        self.embedding_list = torch.nn.ModuleList([torch.nn.Embedding(dim, embed_dim) for dim in field_dims])
        self.linear = torch.nn.Linear(self.n, 1, bias=True)
        self.v = torch.nn.Parameter(torch.Tensor(self.n, self.k))
        torch.nn.init.xavier_uniform_(self.v.T)

    def forward(self, X):
        all_emb = torch.cat([embedding(X[:, i]) for i, embedding in enumerate(self.embedding_list)], dim=1)
        linear_part = self.linear(all_emb)
        square_of_sum = torch.mm(all_emb, self.v) * torch.mm(all_emb, self.v)
        sum_of_square = torch.mm(all_emb * all_emb, self.v*self.v)
        iteraction_part = 0.5 * torch.sum((square_of_sum - sum_of_square), dim=-1, keepdim=True)
        print(iteraction_part.shape)
        logit = linear_part + iteraction_part
        return torch.sigmoid(logit)

from utils import Trainer
EMBEDDING_DIM = 8
LEARNING_RATE = 1e-4
REGULARIZATION = 1e-6
BATCH_SIZE = 4096
EPOCH = 1000
TRIAL = 100
K = 8

mm = model(field_dims, K, EMBEDDING_DIM).to(device)
optimizer = torch.optim.Adam(mm.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION)
loss = torch.nn.BCELoss()

trainer = Trainer(mm, optimizer, loss, BATCH_SIZE)

trainer.train(train_X, train_y, EPOCH, TRIAL, valid_X, valid_y)

test_loss, test_acu = trainer.test(test_X, test_y)
print('test_loss:  {:.5f} | test_auc:  {:.5f}'.format(test_loss, test_acu))

#train_loss: 0.41062 | train_metric: 0.81474
#valid_loss: 0.50407 | valid_metric: 0.71472
#test_loss:  0.48270 | test_auc:  0.72701