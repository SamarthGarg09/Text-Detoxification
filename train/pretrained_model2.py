#%%
# dependencies
import argparse
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5EncoderModel
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb
import os
import logging
import shutil
from collections import defaultdict
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from train.early_stopping import EarlyStopping
from simple_evaluator import Evaluator
from torch.nn.functional import cosine_similarity

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import warnings
warnings.filterwarnings('ignore')

# ignore transformers warnings
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

def setup(rank, world_size):
    "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()


#%%

def load_tokenized_data():
# Load the Civil Comments dataset
    dataset = load_from_disk("/Data/deeksha/disha/code_p/style_transformer_repl/civil_comments_ds")

    # load the T5 tokenizer, the T5 model 
    tokenizer = AutoTokenizer.from_pretrained('ceshine/t5-paraphrase-paws-msrp-opinosis')


    # add normal: and toxic: to the tokenizer as special tokens
    # tokenizer.add_special_tokens({'additional_special_tokens': ['normal:', 'toxic:']})

    dataset['train'] = dataset['train'].shuffle().select(range(10000))
    dataset['test'] = dataset['test'].shuffle().select(range(1000))

    tokenized_train_dataset = dataset['train']
    tokenized_test_dataset = dataset['test']
    # set format to pytorch
    tokenized_train_dataset.set_format('torch')
    tokenized_test_dataset.set_format('torch')

    # remove the text column
    tokenized_train_dataset = tokenized_train_dataset.remove_columns(['text'])
    tokenized_test_dataset = tokenized_test_dataset.remove_columns(['text'])

    return tokenized_train_dataset, tokenized_test_dataset, tokenizer
#%%

def f_step(config, rank, tokenizer, style_model:T5ForConditionalGeneration, disc_model, batch, criterion, style_optimizer, style_scheduler, cyc_rec_weight=0.5, adv_weight=1.0, slf_rec_weight=0.25, copy_penalty_weight=0.5)-> dict:
    """
    Function to perform a single step of training on the style model
    """
    # extract the style codes from the input_ids
    style_code = batch['input_ids'][:, 0:4]
    
    # add 'paraphrase :' to the input_ids
    paraphrase_id = tokenizer.encode('paraphrase')
    paraphrase_id = torch.tensor([paraphrase_id]).to(rank)
    
    # concatenate it with batch['input_ids']
    input_ids = torch.cat([paraphrase_id.repeat(batch['input_ids'].shape[0], 1), batch['input_ids']], dim=1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask.masked_fill_(input_ids == tokenizer.pad_token_id, 0)

    # get the token ids for the style codes and switch the toxic for normal and vie versa
    opposite_style_code = torch.where(style_code == tokenizer.encode('toxic: ', return_tensors='pt').to(style_code.device), tokenizer.encode('normal: ', return_tensors='pt').to(style_code.device), tokenizer.encode('toxic: ', return_tensors='pt').to(style_code.device))
    
    opp_input_ids = torch.cat([opposite_style_code, input_ids[:, 7:]], dim=1)
    opp_input_ids = torch.cat([paraphrase_id.repeat(input_ids.shape[0], 1), opp_input_ids], dim=1)
    opp_attention_mask = torch.ones_like(opp_input_ids)
    opp_attention_mask.masked_fill_(opp_input_ids == tokenizer.pad_token_id, 0)
    '''[paraphrase, opposite_style_code, original_text]'''

    style_optimizer.zero_grad()

    # self_reconstruction_logits = style_model(
    #     input_ids = opp_input_ids,
    #     attention_mask=opp_attention_mask,
    #     labels=opp_input_ids,
    #     return_dict=True,
    # ).logits

    # calculate the loss
    # slf_rec_loss = criterion(self_reconstruction_logits.permute(0, 2, 1), input_ids)
    
    input_embeddings = style_model.module.get_input_embeddings()(input_ids)
    generated_embeddings = style_model.module.get_input_embeddings()(other_class_input_ids)

    # Compute cosine similarity penalty
    copy_penalty = cosine_similarity(input_embeddings, generated_embeddings, dim=-1).mean()


    other_class_input_ids = style_model.module.generate(
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
    labels = torch.where(style_code == tokenizer.encode('toxic: ', return_tensors='pt').to(style_code.device), torch.ones_like(style_code), torch.zeros_like(style_code))
    labels = torch.all(labels == 1, dim=1).long()

    # calculate adversarial loss
    adv_loss = criterion(style_classifier, labels)

    # total loss
    # total_loss = cyc_rec_weight * cyc_rec_loss + adv_weight * adv_loss + slf_rec_weight * slf_rec_loss + copy_penalty_weight * copy_penalty
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
        # 'slf_rec_loss': slf_rec_loss.item(),
    }

# %%

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

# %%

def d_step(config, rank, tokenizer, batch, detoxifier, disc_model, criterion, disc_optimizer, disc_scheduler, disc_type='conditional'):
    '''
    y = fθ(x, s)
    yb = fθ(x,bs)
    Label {x, y} as i 
    Label {yb} as 0 
    Compute loss for dφ
    '''

    disc_optimizer.zero_grad()
    
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    detoxifier.eval()

    # remove style codes from input_ids
    input_ids = input_ids[:, 4:]
    attention_mask = attention_mask[:, 4:]

    # opposite style codes
    style_code = batch['input_ids'][:, :4]
    opposite_style_code =  torch.where(style_code == tokenizer.encode('toxic: ', return_tensors='pt').to(style_code.device), tokenizer.encode('normal: ', return_tensors='pt').to(style_code.device), tokenizer.encode('toxic: ', return_tensors='pt').to(style_code.device))
    opp_input_ids = torch.cat([opposite_style_code, input_ids], dim=1)

    paraphrase_id = tokenizer.encode('paraphrase')
    paraphrase_id = torch.tensor([paraphrase_id]).to(rank)

    opp_input_ids = torch.cat([paraphrase_id.repeat(input_ids.shape[0], 1), opp_input_ids], dim=1)

    opp_attention_mask = torch.ones_like(opp_input_ids)
    opp_attention_mask.masked_fill_(opp_input_ids == tokenizer.pad_token_id, 0)

    # generated the text for the real text [civil: civil sentence]
    with torch.no_grad():
        gen_opp_input_ids = detoxifier.module.generate(
            input_ids = opp_input_ids,
            attention_mask = opp_attention_mask,
            max_length=config['max_length'],
            return_dict=False,
            top_k=4,
            penalty_alpha=0.6,
        )

    gen_opp_attention_mask = torch.ones_like(gen_opp_input_ids)
    gen_opp_attention_mask.masked_fill_(gen_opp_input_ids == tokenizer.pad_token_id, 0)

    # computed logits for the real text [civil: civil sentence]
    logits = disc_model(
        input_ids = gen_opp_input_ids,
        attention_mask = gen_opp_attention_mask,
    )

    # discriminator will predict 1 for real and 0 for generated text
    if disc_type == 'conditional': #fake->0 and real->1
        # check instance of loss as BCEWithLogitsLoss is used for binary classification
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            labels = torch.ones_like(logits)
            real_loss = criterion(logits, labels)
    elif disc_type == "Multi-class": #Toxic->1 and Normal->0
        if isinstance(criterion, nn.CrossEntropyLoss):
            labels = torch.where(opposite_style_code == tokenizer.encode('toxic: ', return_tensors='pt').to(style_code.device), torch.ones_like(style_code), torch.zeros_like(style_code))
            labels = torch.all(labels == 1, dim=1).long()
            real_loss = criterion(logits, labels)

    input_ids = torch.cat([paraphrase_id.repeat(batch['input_ids'].shape[0], 1), batch['input_ids']], dim=1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask.masked_fill_(input_ids == tokenizer.pad_token_id, 0)

    # calculate the logits for the generated text
    with torch.no_grad():
        generated_input_ids = detoxifier.module.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_length=config['max_length'],
            return_dict=False,
            top_k=4,
            penalty_alpha=0.6,
        )
    generated_attention_mask = torch.ones_like(generated_input_ids)
    generated_attention_mask.masked_fill_(generated_input_ids == tokenizer.pad_token_id, 0)

    generated_logits = disc_model(
        input_ids = generated_input_ids,
        attention_mask = generated_attention_mask,
    )

    # fake_labels
    if disc_type == 'conditional':
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            fake_labels = torch.zeros_like(logits)
            fake_loss = criterion(generated_logits, fake_labels)

    elif disc_type == "Multi-class":
        if isinstance(criterion, nn.CrossEntropyLoss):
            labels = torch.where(style_code == tokenizer.encode('toxic: ', return_tensors='pt').to(style_code.device), torch.ones_like(style_code), torch.zeros_like(style_code))
            labels = torch.all(labels == 1, dim=1).long()
            fake_loss = criterion(generated_logits, labels)

    # total loss
    total_loss = real_loss + fake_loss

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(disc_model.parameters(), config['discriminator_max_grad_norm'])
    disc_optimizer.step()
    disc_scheduler.step()
    disc_model.train()

    return {
        'loss': total_loss.item(),
        'real_loss': real_loss.item(),
        'fake_loss': fake_loss.item(),
    }

# %%

def create_dataloaders(train_dataset, test_dataset, batch_size=8, world_size=2):
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size,
        rank=0,
        shuffle=True,
        seed=42
    )
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=0,
        shuffle=False, 
        seed=42
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size//world_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=batch_size//world_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_dataloader, val_dataloader
#%%

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
    logits = detoxifier.module.generate(
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

#%%
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
        rank, world_size,
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
        train_iters.sampler.set_epoch(steps)
        progress_bar = tqdm(range(total_train_steps))
        for batch in batch_iter:
            batch = {k: v.to(rank) for k, v in batch.items()}
            # train the style model
            for _ in range(config['n_detoxifier']):
                style_stats = f_step(
                    config, 
                    rank,
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
                history_stats['slf_rec_loss'].append(style_stats['slf_rec_loss'])
                

            for _ in range(config['n_discriminator']):
                disc_stats = d_step(
                    config, 
                    rank, 
                    tokenizer, 
                    batch,
                    detoxifier,
                    discriminator,
                    disc_criterion,
                    disc_optimizer,
                    disc_scheduler,
                    disc_type,
                )

                history_stats['disc_loss'].append(disc_stats['loss'])
                history_stats['real_loss'].append(disc_stats['fake_loss'])
                history_stats['fake_loss'].append(disc_stats['real_loss'])
            

            progress_bar.update(1)  

            # generate samples    
            steps += 1
            if steps % config['generation_steps'] == 0:
                # log the reference and generated samples into wandb
                test(eval_iters, config, rank, tokenizer, detoxifier, discriminator, evaluator)
        
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
                avg_real_loss = np.mean(history_stats['real_loss'])
                avg_fake_loss = np.mean(history_stats['fake_loss'])

                avg_style_loss = np.mean(history_stats['style_loss'])
                avg_cyc_rec_loss = np.mean(history_stats['cyc_rec_loss'])
                avg_adv_loss = np.mean(history_stats['adv_loss'])
                avg_slf_rec_loss = np.mean(history_stats['slf_rec_loss'])

                result = {
                    'disc_loss': avg_disc_loss,
                    'real_loss': avg_real_loss,
                    'fake_loss': avg_fake_loss,
                    'style_loss': avg_style_loss,
                    'cyc_rec_loss': avg_cyc_rec_loss,
                    'adv_loss': avg_adv_loss,
                    'style_rec_loss': avg_slf_rec_loss,
                }

            # logger.log("Losses and Metrics", result)
                print(f"GPU {rank}:" , result, '\n')
                wandb.log(result)
                history_stats = defaultdict(list)

        epochs += 1
        # early_stopper(avg_style_loss, detoxifier, discriminator)

        # if early_stopper.early_stop:
        #     print("Early stopping")
        #     break

#%%
def test(
    eval_iters,
    config,
    rank, 
    tokenizer,
    detoxifier,
    discriminator,
    evaluator
):
    detoxifier.eval()
    discriminator.eval()

    original_sentences, transferred_sentences, styles = [], [], []
    for batch in tqdm(eval_iters):
        batch = {k: v.to(rank) for k, v in batch.items()}
        # extract style codes from input_ids, attention mask
        style_code = batch['input_ids'][:, :4]
        opposite_style_code = torch.where(style_code == tokenizer.encode('toxic: ', return_tensors='pt').to(style_code.device), tokenizer.encode('normal: ', return_tensors='pt').to(style_code.device), tokenizer.encode('toxic: ', return_tensors='pt').to(style_code.device))
        paraphrase_id = tokenizer.encode('paraphrase')
        paraphrase_id = torch.tensor([paraphrase_id]).to(rank)
        input_ids = torch.cat([paraphrase_id.repeat(batch['input_ids'].shape[0], 1), batch['input_ids']], dim=1)
        attention_mask = torch.ones_like(input_ids)
        attention_mask.masked_fill_(input_ids == tokenizer.pad_token_id, 0)
        
        with torch.no_grad():
            other_class_input_ids = detoxifier.module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_return_sequences=1,
                max_length=config['max_length'],
                top_k=4,
                penalty_alpha=0.6,
                return_dict=False,
            )

        # calculate labels from style codes
        labels = torch.where(style_code == tokenizer.encode('toxic: ', return_tensors='pt').to(style_code.device), torch.ones_like(style_code), torch.zeros_like(style_code))
        labels = torch.all(labels == 1, dim=1).long()  

        original_sentence = tokenizer.batch_decode(batch['input_ids'][:, 4:], skip_special_tokens=True)
        transferred_sentence = tokenizer.batch_decode(other_class_input_ids, skip_special_tokens=True)

        original_sentences.extend(original_sentence)
        transferred_sentences.extend(transferred_sentence)
        styles.extend(tokenizer.batch_decode(batch['input_ids'][:, :4], skip_special_tokens=True))

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


#%%
def main(rank:int, world_size:int, config:dict):
    setup(rank, world_size)
    
    '''
    Input Text : paraphrase normal: Toxic sentence
    '''
    wandb.init(project=config['wandb_project'], name=config['wandb_run_name'])
    tokenized_train_dataset, tokenized_test_dataset, tokenizer = load_tokenized_data()

    train_dataloader, val_dataloader = create_dataloaders(tokenized_train_dataset, tokenized_test_dataset, batch_size=config['batch_size'], world_size=world_size)
    detoxifier = T5ForConditionalGeneration.from_pretrained('ceshine/t5-paraphrase-paws-msrp-opinosis').to(rank)
    discriminator = Discriminator().to(rank)

    # make the vocab size of the discriminator equal to the vocab size of the style model
    discriminator.discriminator.resize_token_embeddings(len(tokenizer))

    detox_ddp = DDP(detoxifier, device_ids=[rank])
    disc_ddp = DDP(discriminator, device_ids=[rank])

    train(
        train_dataloader,
        val_dataloader,
        config,
        tokenizer, 
        detox_ddp,
        disc_ddp,
        None, 
        None,
        rank,
        world_size,
        config['disc_type'],
    )
    cleanup()

# %%

if __name__ == '__main__':
    config = {
    'n_discriminator': 2,
    'n_detoxifier': 1,
    'generation_steps': 500, #100,
    'log_result': 500, #100,
    'save_model': True,
    'path': '/Data/deeksha/disha/code_p/style_transformer_repl/model',
    'disc_type': 'Multi-class',
    'wandb_project': 'Text-Detoxification',
    'wandb_run_name': 'run4',
    'detoxifier_lr': 1e-4,
    'discriminator_lr': 1e-4,
    'batch_size': 1,
    'detoxifier_weight_decay': 0.01,
    'discriminator_weight_decay': 0.01,
    'detoxifier_max_grad_norm': 5,
    'discriminator_max_grad_norm': 5,
    'detoxifier_step_size': 1,
    'discriminator_step_size': 1,
    'max_length': 128,
    'min_length': 20,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_epochs': 2,
    'delta': 0.001,
    'save_steps': 250
    }     # Load your config from a file or define it here

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, config), nprocs=world_size)

