# Report for Graded Assignment 1
> CS-461 Foundation Models and Generative AI
- Mahlia Merville-Hipeau
- Sciper: 345625

## 1. Introduction
This project implements SimCLR for self-supervised visual representation learning on a subset of ImageNet, with the objective of evaluating generalization on both in-distribution and out-of-distribution (OOD) data.

## 2. Data
The dataset follows the assignment specification: 200 classes for training and validation, and 200 unseen classes for OOD evaluation. I trained using augmented images and evaluated using unaugmented versions of each dataset.

## 3. Data Augmentation
I followed the SimCLR paper *“A Simple Framework for Contrastive Learning of Visual Representations”* for the augmentation pipeline. The final transformations included:
- Random resized cropping with horizontal flips  
- Color jitter  
- Random grayscale  
- Gaussian blur  

I experimented with additional augmentations such as equalize, solarize, and Sobel filters, but they decreased performance and training stability, thus I went back to the simpler transform.

## 4. Model Architecture
The model followed the SimCLR framework with three main components:
- **Encoder:** ResNet-18 backbone initialized from scratch, modified for 64×64 inputs by changing the first convolution layer and removing max pooling and the classification head.  
- **Feature layer:** Linear layer mapping the 512-dimensional encoder output to a 1000-dimensional feature vector.  
- **Projector:** Two linear layers with ReLU activation. I tested a simpler single-layer projector but found it less effective.

## 5. Training Configuration
The contrastive loss used was the normalized temperature-scaled cross-entropy (NT-Xent) loss. A temperature parameter of τ = 0.1 provided better results than the default 0.5.  

Key hyperparameters:
- **Learning rate:** `lr = 0.3 × (batch_size / 256)`  
- **Weight decay:** `1e-4`  
- **Optimizer:** SGD  
- **Scheduler:** Cosine annealing with linear warmup  

I first experimented with Adam and StepLR but later adopted SGD with cosine annealing for smoother convergence and better representation quality. LARS, used in the original paper, was avoided for simplicity.

## 6. Evaluation
Two probing methods were used to assess representation quality:
- **k-Nearest Neighbors (k-NN) probe** for unsupervised evaluation.  
- **Linear probe** for a supervised test of feature quality.

Evaluation was conducted on both in-distribution and OOD datasets using non-augmented images. I tracked loss and probe accuracies throughout training to monitor stability and generalization.

## 7. Results
The SimCLR model successfully learned meaningful representations from the 100k training images. The **linear probe** achieved around 62% accuracy on the in-distribution validation set and 45% on OOD data, confirming partial generalization. The **k-NN probe** followed a similar trend, with a 5–10% lower accuracy. While performance on unseen classes was lower, the representations remained semantically consistent. The results suggest that the model effectively captured transferable features despite limited class overlap and reduced dataset size.

## 8. Summary
Implementing SimCLR from scratch provided insight into how design choices in augmentations, architecture, and learning rate scheduling influence self-supervised learning performance. The chosen configuration balanced stability, simplicity, and generalization effectively.
