#%%
# dependencies
from datasets import concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch

#%%
dataset = load_dataset('civil_comments')

toxic_dataset = dataset['train'].filter(lambda example: example['toxicity'] > 0.3)

# for normal text toxicity, severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit are all 0
def filter_normal(example):
    if example['toxicity'] == 0 and example['severe_toxicity'] == 0 and example['obscene'] == 0 and example['threat'] == 0 and example['insult'] == 0 and example['identity_attack'] == 0 and example['sexual_explicit'] == 0:
        return True
    return False

normal_dataset = dataset['train'].filter(filter_normal)

# Print information about the dataset
print(toxic_dataset)
print(normal_dataset)

# select 25000 examples from the toxic and normal dataset
toxic_dataset = toxic_dataset.select(range(5000))
normal_dataset = normal_dataset.select(range(5000))

# add prefixes normal: to the toxic dataset and toxic: to the normal dataset
toxic_dataset = toxic_dataset.map(lambda example: {'text': 'toxic: ' + example['text']})
normal_dataset = normal_dataset.map(lambda example: {'text': 'normal: ' + example['text']})


# combine the 2 datasets
dataset = concatenate_datasets([toxic_dataset, normal_dataset])

def prepare_labels(example):
    labels = []
    if example['text'].startswith('toxic'):
        labels.append(0)
    else:
        labels.append(1)
    return {'labels': labels}

# assign labels to the dataset other than using map method
dataset = dataset.map(prepare_labels)

# remove "toxic" and normal from the text column
dataset = dataset.map(lambda example: {'text': example['text'].split(': ')[1]})

# shuffle the dataset
dataset = dataset.shuffle()

#remove all columns except the text column
dataset = dataset.remove_columns(['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit'])

train_dataset = dataset.train_test_split(test_size=0.2)
test_dataset = train_dataset['test']

# print some information about the datasets
print(train_dataset)
print(test_dataset)
#%%
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_ckpt = 'roberta-base'
# load the T5 tokenizer, the T5 model 
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)

#%%

def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, return_tensors='pt')

# tokenize the train and test datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
#%%
from datasets import Dataset
def remove_short_sentence(ds):
    valid_samples = []
    for example in ds:
        if len(example['text'].split()) > 20:
            valid_samples.append(example)
    del ds
    dataset = Dataset.from_list(valid_samples)

    return dataset

tokenized_train_dataset['train'] = remove_short_sentence(tokenized_train_dataset['train'])
tokenized_train_dataset['test'] = remove_short_sentence(tokenized_train_dataset['test'])
#%%
tokenized_train_dataset = tokenized_train_dataset.remove_columns(['text'])

#%%
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
#%%
# set the training arguments
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    fp16=True,                       # fp16 training
    evaluation_strategy='steps',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,
    report_to=None,
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=args,                           # training arguments, defined above
    train_dataset=tokenized_train_dataset['train'],         # training dataset
    eval_dataset=tokenized_train_dataset['test'],             # evaluation dataset
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
# %%

trainer.train()
# %%

trainer.evaluate()
# %%

# save the model
trainer.save_model('./model')
# %%
