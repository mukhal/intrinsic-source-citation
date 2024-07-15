import datasets
from torch.utils.data import DataLoader
import torch

class LMDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenized_dataset = dataset.map(lambda x: self.tokenize(x), batched=True, remove_columns=dataset.column_names)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.tokenized_dataset[idx]
    
    def tokenize(self, examples):
        '''
        tokenize text, tokenixer url, and concatenate them.
        '''
            # Remove empty lines
        examples['text'] = [
            line for line in examples['text'] if len(line) > 0 and not line.isspace()
        ]
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )