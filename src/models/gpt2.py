import torch as t
from transformers import GPT2Model, GPT2Tokenizer

class GPT2Classifier(t.nn.Module):
    def __init__(self, configs, device='cpu') -> None:
        super().__init__()

        self.device = device
        self.configs = configs

        task_a_loss_weights = t.FloatTensor([1, 3]).to(device)
        self.loss_a = t.nn.CrossEntropyLoss(weight=task_a_loss_weights)
        self.label2idx_a = self.get_label_index_a()
        self.idx2label_a = {v: k for k, v in self.label2idx_a.items()}

        task_b_loss_weights = t.FloatTensor([0.2, 1, 1, 1, 1]).to(device)
        self.loss_b = t.nn.CrossEntropyLoss(weight=task_b_loss_weights)
        self.label2idx_b = self.get_label_index_b()
        self.idx2label_b = {v: k for k, v in self.label2idx_b.items()}

        task_c_loss_weights = t.FloatTensor([0.2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).to(device)
        self.loss_c = t.nn.CrossEntropyLoss(weight=task_c_loss_weights)
        self.label2idx_c = self.get_label_index_c()
        self.idx2label_c = {v: k for k, v in self.label2idx_c.items()}

        self.gpt2 = GPT2Model.from_pretrained(configs.model.gpt2.name).to(device)
        self.head_a = t.nn.Linear(self.gpt2.config.hidden_size, len(self.label2idx_a)).to(device)
        self.head_b = t.nn.Linear(self.gpt2.config.hidden_size, len(self.label2idx_b)).to(device)
        self.head_c = t.nn.Linear(self.gpt2.config.hidden_size, len(self.label2idx_c)).to(device)

        



        


