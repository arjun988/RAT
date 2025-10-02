import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

class EnhancedGPTRATTrainer:
    def __init__(self, model, tokenizer, lr=5e-4, warmup_steps=1000, max_steps=100000, weight_decay=0.1, grad_clip=1.0, device=None, accum_steps=1):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.accum_steps = accum_steps
        self.step_count = 0
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.95), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=lr, total_steps=max_steps, pct_start=min(0.1, warmup_steps/max_steps), anneal_strategy='cos')
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    def train_step(self, input_seq, target_seq):
        self.model.train()
        input_seq = input_seq.to(self.device)
        target_seq = target_seq.to(self.device)
        logits = self.model(input_seq)
        loss = self.criterion(logits.view(-1, self.model.vocab_size), target_seq.view(-1))
        loss = loss / self.accum_steps
        loss.backward()
        if (self.step_count + 1) % self.accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
        self.step_count += 1
        return {"loss": loss.item() * self.accum_steps, "learning_rate": self.scheduler.get_last_lr()[0], "step": self.step_count}
