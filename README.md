# Project: Converting Hand-Written Equations to LaTeX Expressions

## Introduction: 
This project aims to transform handwritten mathematical equations into LaTeX format using advanced machine learning techniques. By leveraging a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) with attention mechanisms, we strive to accurately interpret and digitize complex mathematical expressions. This solution is particularly beneficial in educational and research settings where quick digitization of handwritten notes is required. Our implementation is carried out in Google Colab, providing a flexible and accessible platform for experimentation and demonstration.

## Data: 
The dataset used in our project was obtained from the ”ICFHR 2016 CROHME: Competition on Recognition of Online Handwritten Mathematical Expressions”. This dataset consists of handwritten mathematical expression images in BMP format, along with their corresponding LaTeX captions stored in TXT files. The dataset is divided into two parts: a training set containing 8,836 samples and a test set comprising 987 samples.

Dataset can be downloaded from the GitHub repository: https://github.com/JianshuZhang/WAP/ tree/master/data

## Technical Overview: 
The primary technical challenge is the accurate interpretation of handwritten mathematical symbols, which vary widely in style and complexity. Our approach involves two main stages:
### Model Implementation:
We first experiment with various neural network architectures to find the most effective baseline for recognizing and interpreting the structure of handwritten equations. This includes:

**ResNet50 with positional embedding + LSTM:** Combining ResNet50 for robust feature extraction from images with LSTM for sequence prediction.

**DenseNet with positional embedding + GRU:** Using DenseNet for a denser feature perspective and GRU for handling sequence dependencies with fewer parameters than LSTM.

**CNN with positional embedding + Bi-LSTM with Attention:** Introducing attention network to handle sequences where the order of elements is crucial for correct interpretation.

### Hyperparameter Tuning:

In the second phase, we optimize the chosen model in stage one by adjusting hyper-parameters to enhance performance. This includes tuning various hyper-parameters of the Trainer with random search, and experimenting with different configurations of beam sizes for beam search decoding.

**Random Search- on Trainer**

 "emb_dim": [50, 80, 100],
 
"dec_rnn_h": [256, 512, 1024],

 "dropout": [0.0, 0.1, 0.2],
 
 "lr": [5e-3,1e-3,5e-4, 1e-4],
 
 "lr_decay": [0.1, 0.5, 0.9],
 
 "lr_patience": [2, 3, 4],
 
 "clip": [1.0, 2.0, 3.0]
 
**Grid Search- on Latex Generator**

“Beam size”: [1,3,5,10] (1 is Greedy Search)


## Implementation and Experiment 

The project is implemented across two Colab notebooks:

### Model Architecture Implementation: 

The first notebook details the setup, training, and initial evaluation of different model configurations, and the result is as below. We selected the model of CNN with positional embedding + Bi-LSTM with Attention for futher hyper-parameter tuning.

![Image](https://github.com/users/zachwu123/projects/1/assets/166083422/900eb61f-ace0-4a3a-9597-3375b38770ff)

![Image](https://github.com/users/zachwu123/projects/1/assets/166083422/08ceebb1-dc3f-4c23-b9c2-e8c6312b2e5e)

### Hyper-parameter Tuning: 

The second notebook focuses on refining the models by adjusting hyperparameters to optimize performance, and the result is as below. We first conducted random search to find the best hyper-parameters for the Trainer. 

![Image](https://github.com/users/zachwu123/projects/1/assets/166083422/67921e0c-adcf-4278-9574-2f755e7ba210)

![Impact of Hyper-Parameters on Best Validation Loss](https://github.com/zachwu123/image2latex/assets/166083422/1b25295d-a38b-48a0-96f0-29897ef184bc)

Then, grid search is used to explore the difference between greedy search and beam search of different size. We find out that beam size of 5 generally do the best job. Also as the beam size increases, the computation and time required also generally increase. 

![Edit Scores by Beam Size](https://github.com/zachwu123/image2latex/assets/166083422/f813b00e-30d6-447a-af26-c2b20fda329f)
![BLEU-4 scores by Beam Size](https://github.com/zachwu123/image2latex/assets/166083422/9584eaef-f0b3-48ff-a6cd-d5c2f1fe6fcd)
![Partial Match by Beam Size](https://github.com/zachwu123/image2latex/assets/166083422/4e1a8007-4777-4f3a-9075-ca19dcdeed8d)
![Evaluation Time by Beam Size](https://github.com/zachwu123/image2latex/assets/166083422/712d778d-179e-4f1c-9f03-654c0891f329)

### Training the best model

After selecting the best architecture and hyper-parameter tuning, we trained the model using the best config with batch size of 16 and epoch of 50. The result is shown below. As we can see, the model starts to overfit starting at the 11th epoch.  

![Image](https://github.com/users/zachwu123/projects/1/assets/166083422/206c5fc9-d795-458e-8b2c-0d0459c72edb)

When evaluating, the beam size of 5 has the best partial accuracy of 87.5%

![Image](https://github.com/users/zachwu123/projects/1/assets/166083422/99447130-270f-489d-82c8-209b90e8d848)

## Future Work

Initial results indicate that the attention mechanism significantly improves performance, especially when combined with positional encoding. Future work will focus on further refining these models, exploring additional architectures like combining DesNet with Bi-LSTM, more complex attention mechanism like transformer based attention (or multi-head attention). Given that multi-head attention is computationally expensive, especially for very long sequences, researching sparse versions of Transformer attention, which reduce the complexity per layer, can be beneficial.

The labeled data used to train this model is very limited, with 8k samples only. Therefore, we can see that even the best label become overfitting easily. The next step is to conduct data augmentation or find bigger data (with at least 200k samples) to train the data.
