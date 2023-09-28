# Text-Detoxification 🧼✨
Transform hate speech into civil and respectful language using advanced GANs techniques.

## Introduction 📖
The Text-Detoxification project aims to detoxify hate speech, making the internet a safer and more inclusive space. Leveraging GANs objective functions like cyclic consistency loss, adversarial losses, and the pretrained T5-model with style codes for each label, the project ensures high-quality results. Additionally, techniques like contrastive search decoding and curriculum learning are employed, given the notorious instability of GANs.

## Dataset 📊
The project utilizes the [Civil Comments dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data), also known as the Jigsaw Unintended Bias dataset. This dataset provides a comprehensive collection of comments, aiding in the training and evaluation of the model.

<!-- draw file structure -->
## Folder Structure 📁

```
Text-Detoxification/
├── train/ # Training scripts
├── evaluation/ # Evaluation scripts
├── civil_comments_eda.ipynb
├── requirements.txt # Required packages
└── README.md # This file

```

## Installation 🛠️
1. Clone the repository:
   ```bash
   git clone https://github.com/SamarthGarg09/Text-Detoxification.git

2. Enter the directory:
    ```bash
    cd Text-Detoxification
3. To preprocess the Dataset you can modify:
    ```bash
    cd train
    python preprocess.py

4. Install the required packages:
    ```bash
    pip install -r requirements.txt

5. Train the Model on Multi-Gpu Enviroment:
    ```bash
    cd train
    CUDA_VISIBLE_DEVICES=0,1 python train_multiple_gpu.py

6. Train the model on single GPU:
    ```bash
    cd train
    python train_single_gpu.py
    ```

## Results 📈

Our model achieved promising results in various evaluation metrics:

- **Semantic Similarity**: The model maintained a semantic similarity of **72.01%**, ensuring that the detoxified text retains the original meaning.
  
- **Style Transfer Accuracy**: With an accuracy of **81.55%**, the model effectively transfers the style of the text from hate speech to civil speech.
  
- **Perplexity**: The model achieved a perplexity score of **21.23**, indicating its ability to predict the next word in a sequence with high precision.

These results highlight the effectiveness of our approach in detoxifying hate speech while preserving the core message of the text.

