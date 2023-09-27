#%%
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import torch
from tqdm import tqdm

perplexity_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

#%%
def perplexity(model, encodings, device):
    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)
    model.to(device)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()
# %%

class Evaluator:
    def __init__(self, model_ckpt, src_txt_path, tgt_txt_path, style_model_ckpt):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.sim_model = SentenceTransformer('roberta-base')
        
        with open(src_txt_path, 'r') as f:
            self.orig_texts = f.readlines()
        with open(tgt_txt_path, 'r') as f:
            self.tgt_texts = f.readlines()

        # style transformer model and tokenizer
        self.style_tokenizer = AutoTokenizer.from_pretrained('t5-base')
        self.style_model = T5ForConditionalGeneration.from_pretrained('t5-base').to('cuda')
        # self.style_model = GPT2LMHeadModel.from_pretrained('gpt2').to('cuda')
        # self.style_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        # load model from style_model_ckpt
        self.style_model.load_state_dict(torch.load(style_model_ckpt))

        self.encodings = self.style_tokenizer("\n\n".join(perplexity_ds["text"]), return_tensors="pt")
    def extract_true_labels(self): # 0 -> toxic and 1-> normal
        labels, sents = [], []
        for i in range(len(self.orig_texts)):
            sent = self.orig_texts[i].split(': ')[1]
            if self.orig_texts[i].startswith('toxic'):    
                labels.append(0)
                sents.append(sent)
            else:
                labels.append(1)
                sents.append(sent)
        return labels

    def __call__(self):
        actual_label = self.extract_true_labels()
        tok_texts = self.tokenizer(self.tgt_texts, padding=True, truncation=True, return_tensors="pt")
        # batch to cuda
        # tok_texts = {k: v.to('cuda') for k, v in tok_texts.items()}
        style_logits = self.model(**tok_texts)
        style_logits = style_logits.logits
        style_probs = torch.softmax(style_logits, dim=1)
        pred = torch.argmax(style_probs, dim=1)
        
        # calculate accuracy
        pred = pred.cpu().numpy()
        accuracy = accuracy_score(actual_label, pred)

        # calculate similarity between original and transferred text
        original_embeds = self.sim_model.encode(self.orig_texts, convert_to_tensor=True)
        transferred_embeds = self.sim_model.encode(self.tgt_texts, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(original_embeds, transferred_embeds)
        avg_similarity_score = torch.mean(cosine_similarity).item()

        # perplexity
        # tok_text = self.style_tokenizer(self.orig_texts, padding=True, truncation=True, return_tensors="pt")
        # decoder_sos_token_id = 0
        # tok_text['decoder_input_ids'] = torch.ones(tok_text['input_ids'].shape[0], dtype=torch.long).unsqueeze(-1) * decoder_sos_token_id
        # with torch.no_grad():
        #     logits = self.style_model(**tok_text).logits
        # probs = torch.nn.functional.softmax(logits, dim=-1)
        # perplexity = 2 ** (-torch.log2(probs[0][tok_text['input_ids'].view(-1)]).mean())
        perplexity_score = perplexity(self.style_model, self.encodings, 'cuda')

        return accuracy, avg_similarity_score, perplexity_score
    
eval = Evaluator(
    'roberta-base', 
    '/Data/deeksha/disha/code_p/style_transformer_repl/evaluation/test/o_text.txt', 
    '/Data/deeksha/disha/code_p/style_transformer_repl/evaluation/test/r_text.txt',
    "/Data/deeksha/disha/code_p/style_transformer_repl/model/pytorch_model.bin"
    )
print(eval())



#%%

# %%
# sentence = '''
# normal: Well here we go again. Let's continue to subsidize the cost of power, costs that are out of control due to decisions made by the Liberal government. How nice to know the government has such contempt for the citizens of Ontario that it thinks we won't understand we are being bribed with out own tax dollars - tax dollars we will have to pay back with interest - to make re-election of these corrupt fools possible. Not enough that the deficit remains large, we will now increase that deficit to buy votes. This stinks, just like the power plant cancellation last election. Ontario voters, wake up and throw these bums out!!	
# '''
# tokenizer = AutoTokenizer.from_pretrained('t5-base')
# tokenizer.add_special_tokens({'additional_special_tokens': ['normal:', 'toxic:']})
# tok_text = tokenizer([sentence], padding=True, truncation=True, return_tensors="pt")
# decoder_sos_token_id = 0
# # toxic token id
# # tok_text['decoder_input_ids'] = torch.ones(tok_text['input_ids'].shape[0], dtype=torch.long).unsqueeze(-1) * tokenizer.convert_tokens_to_ids('toxic:')
# tok_text = {k: v.to('cuda') for k, v in tok_text.items()}
# transferred_text1 = T5ForConditionalGeneration.from_pretrained('t5-base').to('cuda').generate(
#     input_ids=tok_text['input_ids'],
#     attention_mask=tok_text['attention_mask'],
#     # decoder_input_ids=tok_text['decoder_input_ids'],
#     max_length=300,
#     num_beams=5,
# )
# transferred_text2 = eval.style_model.generate(
#     input_ids=tok_text['input_ids'],
#     attention_mask=tok_text['attention_mask'],
#     # decoder_input_ids=tok_text['decoder_input_ids'],

#     max_length=300,
#     num_beams=5,
# )
# print(tokenizer.decode(transferred_text1[0], skip_special_tokens=True))
# print(tokenizer.decode(transferred_text2[0], skip_special_tokens=True))

#%%
# from datasets import load_from_disk
# from torch.utils.data import DataLoader

# style_tokenizer = AutoTokenizer.from_pretrained('t5-base')
# style_tokenizer.add_special_tokens({'additional_special_tokens': ['normal:', 'toxic:']})
# test_ds = load_from_disk("/Data/deeksha/disha/code_p/style_transformer_repl/civil_comments_ds")['test']
# style_model = T5ForConditionalGeneration.from_pretrained('t5-base').to('cuda')
# style_model.load_state_dict(torch.load("/Data/deeksha/disha/code_p/style_transformer_repl/model/pytorch_model.bin"))
# # take 5000 samples only
# test_ds = test_ds.select(range(500))
# # generate text for the test dataset from eval.style_model and save it to a file /Data/deeksha/disha/code_p/style_transformer_repl/evaluation/test/predicted_text.txt
# test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)

# style_model.eval()
# style_model.to('cuda')

# for batch in tqdm(test_dl):
#     tok_text = style_tokenizer(batch['text'], padding=True, truncation=True, return_tensors="pt")
#     tok_text = {k: v.to('cuda') for k, v in tok_text.items()}
#     transferred_text = style_model.generate(
#         input_ids=tok_text['input_ids'],
#         attention_mask=tok_text['attention_mask'],
#         # decoder_input_ids=tok_text['decoder_input_ids'],
#         max_length=300,
#         num_beams=5,
#     )
#     # save the original text in "evaluation/test/o_text.txt"
#     with open('/Data/deeksha/disha/code_p/style_transformer_repl/evaluation/test/o_text.txt', 'a') as f:
#         for text in batch['text']:
#             f.write(text + '\n\n\n\n\n\n\n\n\n\n')
#     with open('/Data/deeksha/disha/code_p/style_transformer_repl/evaluation/test/predicted_text.txt', 'a') as f:
#         for text in transferred_text:
#             f.write(style_tokenizer.decode(text, skip_special_tokens=True) + '\n\n\n\n\n\n\n\n\n\n')

# %%
