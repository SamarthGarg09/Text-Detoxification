# !pip install sentencepiece
# !pip install transformers --upgrade
# !pip install tokenizers --upgrade
# !pip install --no-cache-dir transformers sentencepiece datasets wandb

from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5EncoderModel
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb
import os
import logging
from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from train.early_stopping import EarlyStopping
# from simple_evaluator import Evaluator
from torch.nn.functional import cosine_similarity

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings
warnings.filterwarnings('ignore')

# ignore transformers warnings
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
import os

import pandas as pd

def load_parquet_data(file_path):
    """
    Load data from a Parquet file and return it as a pandas DataFrame.
    """
    return pd.read_parquet(file_path)

df_train = load_parquet_data("/kaggle/input/yelp-transformed/yelp_transformed/train.parquet")
df_dev = load_parquet_data("/kaggle/input/yelp-transformed/yelp_transformed/dev.parquet")
df_test = load_parquet_data("/kaggle/input/yelp-transformed/yelp_transformed/test.parquet")

# df_train.head()

train_parquet_path = "/kaggle/input/yelp-transformed/yelp_transformed/train.parquet"
test_parquet_path = "/kaggle/input/yelp-transformed/yelp_transformed/test.parquet"

def load_tokenized_data_from_parquet(train_parquet_path, test_parquet_path):
    """
    Load and tokenize data from Parquet files.
    """
    # Load Parquet files into DataFrames
    train_df = pd.read_parquet(train_parquet_path)
    test_df = pd.read_parquet(test_parquet_path)

    # Load the T5 tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('ceshine/t5-paraphrase-paws-msrp-opinosis', use_fast=False)

    # Assuming the DataFrame has a 'text' column to tokenize and a 'label' column as target
    # Tokenize the text column
    train_encodings = tokenizer(train_df['sentence'].tolist(), truncation=True, padding=True, return_tensors="pt")
    test_encodings = tokenizer(test_df['sentence'].tolist(), truncation=True, padding=True, return_tensors="pt")

    # # Convert labels to tensor
    # train_labels = torch.tensor(train_df['label'].tolist())
    # test_labels = torch.tensor(test_df['label'].tolist())

    # Create a dataset format similar to what was previously returned
    tokenized_train_dataset = {'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask']}
    tokenized_test_dataset = {'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']}

    return tokenized_train_dataset, tokenized_test_dataset, tokenizer

# train_dataset, test_dataset, tokenizer = load_tokenized_data_from_parquet(train_parquet_path, test_parquet_path)
# train_dataset

def f_step(config, rank, tokenizer, style_model:T5ForConditionalGeneration, disc_model, batch, criterion, style_optimizer, style_scheduler, cyc_rec_weight=0.5, adv_weight=1.0, slf_rec_weight=0.25, copy_penalty_weight=0.5)-> dict:
    """
    Function to perform a single step of training on the style model
    """

    separator_token_id = tokenizer.encode('</s>', add_special_tokens=False)[0]
    end_indices = (batch['input_ids'] == separator_token_id).max(dim=1).indices

    # Extract style codes and find the maximum length among them
    style_codes = [batch['input_ids'][i, :idx] for i, idx in enumerate(end_indices)]
    max_style_length = max([code.size(0) for code in style_codes])

    # Pad each style code to the maximum length
    padded_style_codes = []
    for code in style_codes:
        padding_length = max_style_length - code.size(0)
        padding = torch.full((padding_length,), tokenizer.pad_token_id, dtype=torch.long).to(rank)
        padded_code = torch.cat([code, padding])
        padded_style_codes.append(padded_code)

    # Now stack the padded style codes
    style_code = torch.stack(padded_style_codes)

    # add 'paraphrase :' to the input_ids
    paraphrase_id = tokenizer.encode('paraphrase')
    paraphrase_id = torch.tensor([paraphrase_id]).to(rank)

    # concatenate it with batch['input_ids']
    input_ids = torch.cat([paraphrase_id.repeat(batch['input_ids'].shape[0], 1), batch['input_ids']], dim=1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask.masked_fill_(input_ids == tokenizer.pad_token_id, 0)

    # Ensure style codes being compared are of the same length
    neutral_style_code = tokenizer.encode('make this sentence sound neutral: ', return_tensors='pt').to(style_code.device)
    detoxify_style_code = tokenizer.encode('detoxify this sentence: ', return_tensors='pt').to(style_code.device)

    # Pads shorter style codes to match the size of the longer one
    max_length = max(neutral_style_code.size(1), detoxify_style_code.size(1), style_code.size(1))
    neutral_style_code = torch.cat([neutral_style_code, torch.full((1, max_length - neutral_style_code.size(1)), tokenizer.pad_token_id).to(neutral_style_code.device)], dim=1)
    detoxify_style_code = torch.cat([detoxify_style_code, torch.full((1, max_length - detoxify_style_code.size(1)), tokenizer.pad_token_id).to(detoxify_style_code.device)], dim=1)
    style_code = torch.cat([style_code, torch.full((style_code.size(0), max_length - style_code.size(1)), tokenizer.pad_token_id).to(style_code.device)], dim=1)

    opposite_style_code = torch.where(style_code == neutral_style_code, detoxify_style_code, neutral_style_code)

    # Adjust slicing to be dynamic based on end_indices
    opp_input_ids = torch.cat([opposite_style_code, input_ids[:, end_indices[0]:]], dim=1)
    opp_input_ids = torch.cat([paraphrase_id.repeat(input_ids.shape[0], 1), opp_input_ids], dim=1)
    opp_attention_mask = torch.ones_like(opp_input_ids)
    opp_attention_mask.masked_fill_(opp_input_ids == tokenizer.pad_token_id, 0)

    style_optimizer.zero_grad()

    other_class_input_ids = style_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=config['max_length'],
        return_dict=False,
        top_k=4,
        penalty_alpha=0.6
    )
    # add the style code to the input_ids
    other_class_input_ids = torch.cat([opposite_style_code, other_class_input_ids], dim=1)
    other_class_input_ids = torch.cat([paraphrase_id.repeat(input_ids.shape[0], 1), other_class_input_ids], dim=1)
    other_class_attention_mask = torch.ones_like(other_class_input_ids)
    other_class_attention_mask.masked_fill_(other_class_input_ids == tokenizer.pad_token_id, 0)
    '''[paraphrase, opposite_style_code, generated_text]'''

    input_embeddings = style_model.get_input_embeddings()(input_ids)
    generated_embeddings = style_model.get_input_embeddings()(other_class_input_ids)

    # Handle mismatch in embedding sizes by padding the shorter embedding tensor
    max_seq_length = max(input_embeddings.size(1), generated_embeddings.size(1))
    if input_embeddings.size(1) < max_seq_length:
        padding = torch.zeros((input_embeddings.size(0), max_seq_length - input_embeddings.size(1), input_embeddings.size(2))).to(rank)
        input_embeddings = torch.cat([input_embeddings, padding], dim=1)
    elif generated_embeddings.size(1) < max_seq_length:
        padding = torch.zeros((generated_embeddings.size(0), max_seq_length - generated_embeddings.size(1), generated_embeddings.size(2))).to(rank)
        generated_embeddings = torch.cat([generated_embeddings, padding], dim=1)

    # Compute cosine similarity penalty
    copy_penalty = cosine_similarity(input_embeddings, generated_embeddings, dim=-1).mean()

    cyc_rec_loss = style_model(
        input_ids = other_class_input_ids,
        labels = input_ids,
        attention_mask = other_class_attention_mask,
        return_dict = True,
    ).loss

    # adversarial loss
    disc_model.eval()
    other_class_input_ids = other_class_input_ids[:, 7:]
    other_class_attention_mask = other_class_attention_mask[:, 7:]

    with torch.no_grad():
        style_classifier = disc_model(
            input_ids = other_class_input_ids,
            attention_mask = other_class_attention_mask,
        )

    # compute labels
    labels = torch.where(style_code == tokenizer.encode('make this sentence sound neutral: ', return_tensors='pt').to(style_code.device), torch.ones_like(style_code), torch.zeros_like(style_code))
    labels = torch.all(labels == 1, dim=1).long()

    # calculate adversarial loss
    adv_loss = criterion(style_classifier, labels)

    total_loss = cyc_rec_weight * cyc_rec_loss + adv_weight * adv_loss  + copy_penalty_weight * copy_penalty

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(style_model.parameters(), config['detoxifier_max_grad_norm'])
    style_optimizer.step()
    style_scheduler.step()
    style_model.train()

    return {
        'loss': total_loss.item(),
        'cyc_rec_loss': cyc_rec_loss.item(),
        'adv_loss': adv_loss.item(),
    }


## Discriminator Forward Step

class Discriminator(nn.Module):
    def __init__(self, model_ckpt = "ceshine/t5-paraphrase-paws-msrp-opinosis"):
        super().__init__()

        self.discriminator = T5EncoderModel.from_pretrained(model_ckpt)
        self.classifier = nn.Linear(self.discriminator.config.d_model, 2)

    def forward(self, input_ids, attention_mask):
        hidden_states = self.discriminator(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict = True,
        ).last_hidden_state

        logits = self.classifier(hidden_states[:, 0, :])

        return logits


def d_step(config, rank, tokenizer, batch, detoxifier, disc_model, criterion, disc_optimizer, disc_scheduler):

    # Extract the end index of the style code
    separator_token_id = tokenizer.encode('</s>', add_special_tokens=False)[0]
    end_indices = (batch['input_ids'] == separator_token_id).max(dim=1).indices

    # 1. Prepend the paraphrase token
    paraphrase_id = tokenizer.encode('paraphrase')
    paraphrase_id = torch.tensor([paraphrase_id]).to(rank)
    input_ids = torch.cat([paraphrase_id.repeat(batch['input_ids'].shape[0], 1), batch['input_ids']], dim=1)

    attention_mask = torch.ones_like(input_ids)
    attention_mask.masked_fill_(input_ids == tokenizer.pad_token_id, 0)

    # 2. Detoxifier's First Pass (o1)
    with torch.no_grad():
        o1 = detoxifier.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=config['max_length'])
        o1_attention_mask = torch.ones_like(o1)
        o1_attention_mask.masked_fill_(o1 == tokenizer.pad_token_id, 0)

    # 3. Preparation for Detoxifier's Second Pass (o2)
    neutral_encoded = tokenizer.encode('make this sentence sound neutral: ', return_tensors='pt').squeeze().to(rank)
    detoxify_encoded = tokenizer.encode('detoxify this sentence:', return_tensors='pt').squeeze().to(rank)

    opposite_style_code = []
    for idx in end_indices:
        if torch.equal(batch['input_ids'][:, :idx].squeeze(), neutral_encoded):
            opposite_style_code.append(detoxify_encoded)
        else:
            opposite_style_code.append(neutral_encoded)
    opposite_style_code = torch.stack(opposite_style_code)

    opp_input_ids = torch.cat([paraphrase_id.repeat(input_ids.shape[0], 1), opposite_style_code, input_ids[:, end_indices[0]:]], dim=1)
    opp_attention_mask = torch.ones_like(opp_input_ids)
    opp_attention_mask.masked_fill_(opp_input_ids == tokenizer.pad_token_id, 0)

    with torch.no_grad():
        o2 = detoxifier.generate(input_ids=opp_input_ids, attention_mask=opp_attention_mask, max_length=config['max_length'])
        o2_attention_mask = torch.ones_like(o2)
        o2_attention_mask.masked_fill_(o2 == tokenizer.pad_token_id, 0)

    # 4. Label Creation
    labels = []
    for i, idx in enumerate(end_indices):
        if torch.equal(batch['input_ids'][i, :idx].squeeze(), neutral_encoded):
            labels.append(1)
        else:
            labels.append(0)
    labels = torch.tensor(labels).to(rank)

    # 5. Passing Outputs to Discriminator
    logits_o1 = disc_model(
        input_ids=o1,
        attention_mask=o1_attention_mask
    )
    logits_o2 = disc_model(
        input_ids=o2,
        attention_mask=o2_attention_mask
    )

    # 6. Loss Calculation
    loss_o1 = criterion(logits_o1, labels)
    loss_o2 = criterion(logits_o2, 1 - labels)  # using opposite labels for o2
    total_loss = loss_o1 + loss_o2

    # 7. Optimization
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(disc_model.parameters(), config['discriminator_max_grad_norm'])
    disc_optimizer.step()
    disc_scheduler.step()

    return {
        'loss': total_loss.item(),
        'loss_o1': loss_o1.item(),
        'loss_o2': loss_o2.item(),
    }

## Dataloaders

def create_dataloaders(train_dataset, test_dataset, batch_size=8):

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_dataloader, val_dataloader

def generate_samples(
    detoxifier,
    tokenizer,
    input_ids,
    attention_mask,
    num_return_sequences=1,
    max_length=256,
    return_dict=False,
):
    """
    Function to generate samples from the style model
    """
    # generate the logits
    logits = detoxifier.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_return_sequences=num_return_sequences,
        max_length=max_length,
        min_length=config['min_length'],
        top_k=4,
        penalty_alpha=0.6,
        return_dict=return_dict,
    )

    # decode the logits
    generated_samples = tokenizer.batch_decode(logits, skip_special_tokens=True)

    return generated_samples

## Training and Testing the model
# train the model
def train(
        train_iters,
        eval_iters,
        config,
        tokenizer,
        detoxifier,
        discriminator,
        logger,
        evaluator,
        device,
        # rank, world_size,
        disc_type='conditional'
    ):

    # train the discriminator

    disc_criterion = nn.CrossEntropyLoss()
    gen_criterion = nn.CrossEntropyLoss()
    disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=config['discriminator_lr'])
    disc_scheduler = torch.optim.lr_scheduler.StepLR(disc_optimizer, step_size=config['discriminator_step_size'], gamma=0.9)
    style_optimizer = torch.optim.AdamW(detoxifier.parameters(), lr=config['detoxifier_lr'])
    style_scheduler = torch.optim.lr_scheduler.StepLR(style_optimizer, step_size=config['detoxifier_step_size'], gamma=0.9)

    steps, epochs = 0, 0
    batch_iter = iter(train_iters)
    history_stats = defaultdict(list)
    total_train_steps = len(train_iters)
    early_stopper = EarlyStopping(patience=3, verbose=True, delta=config['delta'], path=config['path'])
    while True:
        # train_iters.sampler.set_epoch(steps)
        progress_bar = tqdm(range(total_train_steps))
        for batch in batch_iter:
            batch = {k: v.to(device) for k, v in batch.items()}
            # train the style model
            for _ in range(config['n_detoxifier']):
                style_stats = f_step(
                    config,
                    device,
                    tokenizer,
                    detoxifier,
                    discriminator,
                    batch,
                    gen_criterion,
                    style_optimizer,
                    style_scheduler,
                )

                history_stats['style_loss'].append(style_stats['loss'])
                history_stats['cyc_rec_loss'].append(style_stats['cyc_rec_loss'])
                history_stats['adv_loss'].append(style_stats['adv_loss'])
                # history_stats['slf_rec_loss'].append(style_stats['slf_rec_loss'])


            for _ in range(config['n_discriminator']):
                disc_stats = d_step(
                    config,
                    device,
                    tokenizer,
                    batch,
                    detoxifier,
                    discriminator,
                    disc_criterion,
                    disc_optimizer,
                    disc_scheduler,
                )

                history_stats['disc_loss'].append(disc_stats['loss'])
                history_stats['loss_o1'].append(disc_stats['loss_o1'])
                history_stats['loss_o2'].append(disc_stats['loss_o2'])


            progress_bar.update(1)

            # generate samples
            steps += 1
            if steps % config['generation_steps'] == 0:
                # log the reference and generated samples into wandb
                test(eval_iters, config, device, tokenizer, detoxifier, discriminator, evaluator)

            # save the model
            # if steps % config['save_steps'] == 0:
            #     if os.path.exists(config['path']):
            #         shutil.rmtree(config['path'])
            #     else:
            #         os.makedirs(config['path'])
                # if rank == 0:



            # calculate avg
            if steps % config['log_result'] == 0:
                avg_disc_loss = np.mean(history_stats['disc_loss'])
                avg_real_loss = np.mean(history_stats['loss_o1'])
                avg_fake_loss = np.mean(history_stats['loss_o2'])

                avg_style_loss = np.mean(history_stats['style_loss'])
                avg_cyc_rec_loss = np.mean(history_stats['cyc_rec_loss'])
                avg_adv_loss = np.mean(history_stats['adv_loss'])
#                 avg_slf_rec_loss = np.mean(history_stats['slf_rec_loss'])

                result = {
                    'disc_loss': avg_disc_loss,
                    'real_loss': avg_real_loss,
                    'fake_loss': avg_fake_loss,
                    'style_loss': avg_style_loss,
                    'cyc_rec_loss': avg_cyc_rec_loss,
                    'adv_loss': avg_adv_loss,
#                     'style_rec_loss': avg_slf_rec_loss,
                }

            # logger.log("Losses and Metrics", result)
                print(f"GPU {device}:" , result, '\n')
                wandb.log(result)
                history_stats = defaultdict(list)

        epochs += 1
        # early_stopper(avg_style_loss, detoxifier, discriminator)

        # if early_stopper.early_stop:
        #     print("Early stopping")
        #     break
        del batch
        torch.cuda.empty_cache()

def test(
    eval_iters,
    config,
    device,
    tokenizer,
    detoxifier,
    discriminator,
    evaluator
):
    detoxifier.eval()
    discriminator.eval()

    original_sentences, transferred_sentences, styles = [], [], []
    for batch in tqdm(eval_iters):
        batch = {k: v.to(device) for k, v in batch.items()}
        # extract style codes from input_ids, attention mask
        separator_token_id = tokenizer.encode('</s>', add_special_tokens=False)[0]
        end_indices = (batch['input_ids'] == separator_token_id).max(dim=1).indices

        # Extract style codes and find the maximum length among them
        style_codes = [batch['input_ids'][i, :idx] for i, idx in enumerate(end_indices)]
        max_style_length = max([code.size(0) for code in style_codes])

        # Pad each style code to the maximum length
        padded_style_codes = []
        for code in style_codes:
            padding_length = max_style_length - code.size(0)
            padding = torch.full((padding_length,), tokenizer.pad_token_id, dtype=torch.long).to(device)
            padded_code = torch.cat([code, padding])
            padded_style_codes.append(padded_code)

        # Now stack the padded style codes
        style_code = torch.stack(padded_style_codes)

        # add 'paraphrase :' to the input_ids
        paraphrase_id = tokenizer.encode('paraphrase')
        paraphrase_id = torch.tensor([paraphrase_id]).to(device)

        # concatenate it with batch['input_ids']
        input_ids = torch.cat([paraphrase_id.repeat(batch['input_ids'].shape[0], 1), batch['input_ids']], dim=1)
        attention_mask = torch.ones_like(input_ids)
        attention_mask.masked_fill_(input_ids == tokenizer.pad_token_id, 0)

        # Ensure style codes being compared are of the same length
        neutral_style_code = tokenizer.encode('make this sentence sound neutral: ', return_tensors='pt').to(style_code.device)
        detoxify_style_code = tokenizer.encode('detoxify this sentence: ', return_tensors='pt').to(style_code.device)

        # Pads shorter style codes to match the size of the longer one
        max_length = max(neutral_style_code.size(1), detoxify_style_code.size(1), style_code.size(1))
        neutral_style_code = torch.cat([neutral_style_code, torch.full((1, max_length - neutral_style_code.size(1)), tokenizer.pad_token_id).to(neutral_style_code.device)], dim=1)
        detoxify_style_code = torch.cat([detoxify_style_code, torch.full((1, max_length - detoxify_style_code.size(1)), tokenizer.pad_token_id).to(detoxify_style_code.device)], dim=1)
        style_code = torch.cat([style_code, torch.full((style_code.size(0), max_length - style_code.size(1)), tokenizer.pad_token_id).to(style_code.device)], dim=1)

        with torch.no_grad():
            other_class_input_ids = detoxifier.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_return_sequences=1,
                max_length=config['max_length'],
                top_k=4,
                penalty_alpha=0.6,
                return_dict=False,
            )

        # compute labels
        labels = torch.where(style_code == tokenizer.encode('make this sentence sound neutral: ', return_tensors='pt').to(style_code.device), torch.ones_like(style_code), torch.zeros_like(style_code))
        labels = torch.all(labels == 1, dim=1).long()

        original_sentence = tokenizer.batch_decode(batch['input_ids'][:, 4:], skip_special_tokens=True)
        transferred_sentence = tokenizer.batch_decode(other_class_input_ids, skip_special_tokens=True)

        original_sentences.extend(original_sentence)
        transferred_sentences.extend(transferred_sentence)
        styles.extend(tokenizer.batch_decode(batch['input_ids'][:, :4], skip_special_tokens=True))

        del batch, style_code, paraphrase_id, input_ids, attention_mask
        torch.cuda.empty_cache()

    labels = torch.tensor(np.int8(np.array(styles) == 'toxic:'))
    # similarity = evaluator.similarity(original_sentences, transferred_sentences)
    # style_accuracy = evaluator.style_accuracy(transferred_sentences, labels)
    # print(
    #     f"Similarity: {similarity:.4f}, Style Accuracy: {style_accuracy:.4f}"
    # )
    # wandb.log({
    #     'similarity': similarity,
    #     'style_accuracy': style_accuracy,
    # })
    # log media to wandb
    data = pd.DataFrame({'reference_sentence': original_sentences, 'generated_samples': transferred_sentences, 'styles': styles})
    # print(data)
    wandb.log({"samples": wandb.Table(dataframe=data.sample(10))})

## Configs

config = {
    'n_discriminator': 2,
    'n_detoxifier': 1,
    'generation_steps': 100,
    'log_result': 100,
    'save_model': True,
    'path': 'model',
    'disc_type': 'Multi-class',
    'wandb_project': 'Text-Detoxification',
    'wandb_run_name': 'run4',
    'detoxifier_lr': 1e-4,
    'discriminator_lr': 1e-4,
    'batch_size': 64,
    'detoxifier_weight_decay': 0.01,
    'discriminator_weight_decay': 0.01,
    'detoxifier_max_grad_norm': 5,
    'discriminator_max_grad_norm': 5,
    'detoxifier_step_size': 1,
    'discriminator_step_size': 1,
    'max_length': 32,
    'min_length': 20,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_epochs': 2,
    'delta': 0.001,
    'save_steps': 250
    }     # Load your config from a file or define it here

device = 'cuda'

wandb.login()
wandb.init(project=config['wandb_project'], name=config['wandb_run_name'])
train_dataset, test_dataset, tokenizer = load_tokenized_data_from_parquet(train_parquet_path, test_parquet_path)


detoxifier = T5ForConditionalGeneration.from_pretrained('ceshine/t5-paraphrase-paws-msrp-opinosis').to(device)
discriminator = Discriminator().to(device)

# make the vocab size of the discriminator equal to the vocab size of the style model
discriminator.discriminator.resize_token_embeddings(len(tokenizer))


class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __len__(self):
        # Assumes all items in the dictionary have the same length
        return len(next(iter(self.data_dict.values())))

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data_dict.items()}

train_dataset = CustomDataset(train_dataset)
test_dataset = CustomDataset(test_dataset)

train_dataloader, val_dataloader = create_dataloaders(train_dataset, test_dataset, batch_size=config['batch_size'])

train(
    train_dataloader,
    val_dataloader,
    config,
    tokenizer,
    detoxifier,
    discriminator,
    None,
    None,
    device,
    # rank,
    # world_size,
    config['disc_type'],
)
torch.cuda.empty_cache()