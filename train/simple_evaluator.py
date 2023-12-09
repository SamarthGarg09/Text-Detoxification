import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from torch.nn.functional import cosine_similarity

class SimpleEvaluator(object):
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

def transform_text(detoxifier, tokenizer, sentences, device, max_length=50):
    """
    Transform the original sentences using the detoxifier model.

    Args:
    - detoxifier: The model used for style transfer.
    - tokenizer: Tokenizer for the detoxifier model.
    - sentences: List of original sentences.
    - device: Device on which to run the model (e.g., 'cuda' or 'cpu').
    - max_length: Maximum length for the generated sentences.

    Returns:
    - List of transformed sentences.
    """
    # Prepend the "paraphrase:" token to each sentence
    sentences = [f"paraphrase: {sent}" for sent in sentences]
    
    # Tokenize and generate
    input_ids = tokenizer(sentences, return_tensors='pt', truncation=True, padding=True, max_length=max_length).input_ids.to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    
    with torch.no_grad():
        transformed_ids = detoxifier.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
        
    # Decode the transformed sentences
    transformed_sentences = tokenizer.batch_decode(transformed_ids, skip_special_tokens=True)
    
    return transformed_sentences


def evaluate_style_accuracy(model_name, original_sentences, transformed_sentences, target_style):
    """
    Evaluate the style accuracy of the transformed sentences.

    Args:
    - model_name: Name or path of the pre-trained transformer model.
    - original_sentences: List of original sentences.
    - transformed_sentences: List of transformed sentences.
    - target_style: Target style for the transformed sentences.

    Returns:
    - Style accuracy of the transformed sentences.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    inputs = tokenizer(transformed_sentences, return_tensors='pt', truncation=True, padding=True)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_styles = logits.argmax(dim=1).tolist()
        
    correct_predictions = sum(1 for pred, target in zip(predicted_styles, target_style) if pred == target)
    
    return correct_predictions / len(transformed_sentences)


class Evaluator:
    def __init__(self, detoxifier, tokenizer, device):
        self.detoxifier = detoxifier
        self.tokenizer = tokenizer
        self.device = device
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').eval().to(device)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').eval().to(device)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def get_semantic_similarity(self, original_sentences, transformed_sentences):
        original_embeddings = self.bert_model(self.bert_tokenizer(original_sentences, return_tensors='pt', padding=True, truncation=True).input_ids.to(self.device)).last_hidden_state.mean(dim=1)
        transformed_embeddings = self.bert_model(self.bert_tokenizer(transformed_sentences, return_tensors='pt', padding=True, truncation=True).input_ids.to(self.device)).last_hidden_state.mean(dim=1)
        similarities = cosine_similarity(original_embeddings, transformed_embeddings).mean().item()
        return similarities

    def get_perplexity(self, sentences):
        inputs = self.gpt2_tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            loss = self.gpt2_model(input_ids=inputs.input_ids.to(self.device), attention_mask=inputs.attention_mask.to(self.device), labels=inputs.input_ids.to(self.device)).loss
        return torch.exp(loss).item()

    def evaluate(self, original_sentences, target_style, model_name='bert-base-uncased'):
        transformed_sentences = transform_text(self.detoxifier, self.tokenizer, original_sentences, self.device)
        
        # Style Accuracy
        if model_name:
            style_acc = evaluate_style_accuracy(model_name, original_sentences, transformed_sentences, target_style)
            print(f"Style Accuracy: {style_acc:.4f}")
        
        # Semantic Similarity
        similarity = self.get_semantic_similarity(original_sentences, transformed_sentences)
        print(f"Semantic Similarity: {similarity:.4f}")

        # Perplexity
        perplexity = self.get_perplexity(transformed_sentences)
        print(f"Perplexity: {perplexity:.4f}")

        return style_acc, similarity, perplexity
