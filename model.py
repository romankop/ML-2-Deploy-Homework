from torch.utils.data import Dataset
from transformers.models.distilbert.modeling_distilbert import Transformer
from torch import nn
import torch

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, truncation='longest_first'):

        self.X = torch.tensor([tokenizer.encode(question + ' [SEP] ' + passage,
                              add_special_tokens=True,
                              padding='max_length',
                              truncation=truncation,
                              max_length=512)
                              for question, passage in zip(df.question.values,
                                                          df.passage.values)],
                              dtype=torch.int64)

        self.y = torch.tensor(df.answer.values, dtype=torch.float32)

        return

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        return self.X[idx], self.y[idx]


class QuestAnsweringDistilBERT(nn.Module):

  def __init__(self, bert, config, freeze_type, n_layers=2,
               hidden_dim=3072, dim=768):
    
        super(QuestAnsweringDistilBERT, self).__init__()
               
        self.bert = bert
        self.config = config
        self.config.n_layers = n_layers
        self.config.hidden_dim = hidden_dim
        self.config.dim = dim

        self.heads = Transformer(self.config)
        self.dense = nn.Linear(self.config.dim * self.config.max_position_embeddings,
                               1)

        if freeze_type == 'all':

          for param in self.bert.parameters():
            param.requires_grad = False

        elif freeze_type in ('emb', 'part'):
          
          for param in self.bert.embeddings.parameters():
            param.requires_grad = False

  def forward(self, x):

        part_mask = torch.where(x != 0, torch.tensor(1), torch.tensor(0))

        x = self.bert(x, attention_mask=part_mask).last_hidden_state
        
        head_mask = torch.ones((self.config.n_layers,))
        x = self.heads(x, attn_mask=part_mask, head_mask=head_mask)[0]
        x = self.dense(x.flatten(start_dim=1))
        return x.flatten()