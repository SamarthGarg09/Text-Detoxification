#%%
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer
import numpy as np
import re

# Load the dataset
dataset = load_dataset("civil_comments")

#%%

tokenizer = AutoTokenizer.from_pretrained('ceshine/t5-paraphrase-paws-msrp-opinosis')
def preprocess_text(example):
    # Lowercasing (optional based on the model being used)
    text = example["text"].lower()
    
    # Handling URLs (replacing with [URL] token)
    text = re.sub(r'http\S+', '[URL]', text)
    
    # Handling mentions (replacing with [USER] token)
    text = re.sub(r'@\S+', '[USER]', text)
    
    return example

def mark_sample(example):
    example.setdefault('label', None)
    if example["toxicity"] > 0.8:
        example["label"] = 1
    elif example['toxicity'] == 0 and example['severe_toxicity'] == 0 and example['obscene'] == 0 and example['threat'] == 0 and example['insult'] == 0 and example['identity_attack'] == 0 and example['sexual_explicit'] == 0:
        example["label"] = 0
    return example

#%%

dataset = dataset.map(mark_sample)

# Sort the training dataset based on comment length
sorted_dataset = dataset["train"].sort("text")

# Filter out comments with fewer than 4 tokens
def length_filter(example):

    # length should be at least 4 tokens and at max 30 tokens
    return 4 <= len(example["text"].split()) <= 30

filtered_dataset = sorted_dataset.filter(lambda x: length_filter(x))

#%%

# Decide on the size of the subset
subset_size = 12000  # Example size, can be adjusted

# Sample an equal number of toxic and non-toxic comments
toxic_dataset = filtered_dataset.filter(lambda x: x["label"] == 1).select(range(subset_size // 2))
normal_dataset = filtered_dataset.filter(lambda x: x["label"] == 0).select(range(subset_size // 2))

#%%
toxic_dataset = toxic_dataset.map(preprocess_text)
normal_dataset = normal_dataset.map(preprocess_text)

#%%

toxic_dataset = toxic_dataset.map(lambda example: {'text': 'normal: </s>' + example['text']})
normal_dataset = normal_dataset.map(lambda example: {'text': 'toxic: </s>' + example['text']})


def sort_dataset_by_difficulty(dataset):
    
    # Compute lengths of sentences
    lengths = [len(tokenizer.tokenize(example['text'])) for example in dataset]
    sorted_indices = np.argsort(lengths)
    
    # Sort the dataset by sentence lengths
    sorted_dataset = dataset.select(sorted_indices)
    
    return sorted_dataset

toxic_dataset = sort_dataset_by_difficulty(toxic_dataset)
normal_dataset = sort_dataset_by_difficulty(normal_dataset)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, return_tensors='pt')

toxic_dataset = toxic_dataset.map(tokenize_function, batched=True)
normal_dataset = normal_dataset.map(tokenize_function, batched=True)

# Combine the samples to create the balanced subset
dataset = concatenate_datasets([toxic_dataset, normal_dataset])
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# shuffle the dataset
dataset = dataset.shuffle()

#remove all columns except the text column
dataset = dataset.remove_columns(['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit'])

dataset=DatasetDict({
    'train': dataset['train'],
    'test': dataset['test'],
})

#%%
dataset.save_to_disk("./civil_comments_dataset_main")
print(dataset)

# %%
