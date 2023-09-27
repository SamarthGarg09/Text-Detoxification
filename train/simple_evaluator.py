import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score

class Evaluator(object):
    def __init__(self, st_ckpt):
        self.sim_model = SentenceTransformer('roberta-base')
        self.style_acc_model = AutoModelForSequenceClassification.from_pretrained(st_ckpt, num_labels=2)
        self.style_tokenizer = AutoTokenizer.from_pretrained(st_ckpt)

    def similarity(self, original_sentence, transferred_sentence):
        """Calculates the similarity between the original and transferred sentence using the generator model."""
        original_embeds = self.sim_model.encode(original_sentence, convert_to_tensor=True)
        transferred_embeds = self.sim_model.encode(transferred_sentence, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(original_embeds, transferred_embeds)
        avg_similarity_score = torch.mean(cosine_similarity).item()
        return avg_similarity_score

    def style_accuracy(self, transferred_text, labels):
        """Calculates the style accuracy of the discriminator model."""
        tok_text = self.style_tokenizer(transferred_text, padding=True, truncation=True, return_tensors="pt")
        style_logits = self.style_acc_model(**tok_text).logits
        style_probs = torch.softmax(style_logits, dim=1)
        pred = torch.argmax(style_probs, dim=1)
        pred = pred.cpu().numpy()
        # check if labels not on cpu
        if labels.device != torch.device('cpu'):
            labels = labels.cpu().numpy()
        style_accuracy = accuracy_score(labels, pred)
        return style_accuracy

    def final_metrics(self, original_sentence, transferred_sentence, labels):
        """Calculates the final metrics."""
        similarity = self.similarity(original_sentence, transferred_sentence)
        style_accuracy = self.style_accuracy(original_sentence, labels)
        return similarity, style_accuracy