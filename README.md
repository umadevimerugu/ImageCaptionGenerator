# ImageCaptionGenerator
Image Caption Generator – Project Documentation
Dataset: Flickr8k
Model: CNN Encoder + LSTM Decoder
Domain: Deep Learning / Computer Vision / Natural Language Processing

1. Introduction
Image captioning is the task of generating a meaningful sentence for a given image. It combines computer vision to understand image content and natural language processing to generate descriptive sentences.

This project uses the Flickr8k dataset, which contains 8000 images with 5 captions each.
The goal is to train a CNN‑RNN based encoder–decoder model to automatically generate captions for real‑world images.

2. Project Workflow
The complete pipeline includes:

Dataset Preparation
Caption Preprocessing
Feature Extraction using CNN Encoder
Caption Tokenization
Model Architecture (Encoder–Decoder)
Model Training
Caption Prediction
Evaluation & Testing
3. Dataset Details
Flickr8k Images: 8000 images of people and animals performing activities
Captions: Each image contains 5 human‑written captions
Files Used:
Flickr8k_Dataset (images)
Flickr8k_text:
Flickr8k.token.txt (all captions)
Flickr_8k.trainImages.txt
Flickr_8k.testImages.txt
Flickr_8k.devImages.txt
4. Data Preprocessing
4.1 Caption Cleaning
Steps performed:

Convert text to lowercase
Remove punctuation
Remove numbers
Remove extra spaces
Add startseq and endseq tokens to each caption
This helps the LSTM recognize beginning and end of sentences.

4.2 Vocabulary Creation
Count frequency of all words
Remove rare words (occurrence < 10)
Build final vocabulary used in the model
5. Feature Extraction (Encoder)
A pre‑trained InceptionV3 or VGG16 model is used as the image encoder.

Steps:

Load the pre‑trained CNN model (trained on ImageNet)
Remove the classification layer
Input images → Output 2048‑dimension feature vector
Save extracted features for all images
These features represent the visual content of the image.

6. Caption Tokenization
Using Keras Tokenizer:

Convert words into integer sequences
Padding sequences
Create mapping: word → index and index → word
Determine maximum caption length
7. Model Architecture
Encoder–Decoder Structure
7.1 Encoder (CNN)
Pre-trained InceptionV3 / VGG16
Output: fixed-length feature vector
Dense layer converts to suitable input for LSTM
7.2 Decoder (RNN)
Embedding layer
LSTM network to model caption sequences
Dense layer with softmax activation to predict next words
7.3 Combined Model
Inputs:

Image feature vector
Caption sequence
Output:

Predicted next word in sequence
Mathematical representation:
Given image features 
𝑉
V and caption words 

8. Model Training
Loss function: categorical cross-entropy
Optimizer: Adam
Training runs for 20–30 epochs
Teacher forcing is used: model learns next word based on previous ground‑truth words
Caption sequences used during training:

(startseq),(startseq word1),(startseq word1 word2),...
9. Caption Generation (Inference)
During prediction:

Input test image
Extract features using encoder
Start with "startseq"
Predict next word
Append predicted word
Repeat until "endseq" or max length
Uses greedy search or beam search.

Example output:
"A dog running across a field"

10. Evaluation Metrics
Common evaluation scores:

BLEU‑1
BLEU‑2
BLEU‑3
BLEU‑4
BLEU measures how close the generated captions are to human‑written captions.

11. Results
Typical results for Flickr8k dataset:

BLEU‑1: ~0.55
BLEU‑2: ~0.32
BLEU‑3: ~0.20
BLEU‑4: ~0.10
Model performs well for simple actions and common objects.

12. Applications
Photo‑tagging systems
Visually impaired assistance
Content‑based image retrieval
Social media automation
Robotics / scene understanding
13. Conclusion
The Image Caption Generator demonstrates how combining computer vision and natural language processing allows machines to understand images and describe them in human language.
Using a CNN‑Encoder + LSTM‑Decoder, the model can generate meaningful captions for unseen images.

14. References
Flickr8k Dataset
Keras Documentation
