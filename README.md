# SLR
isolated & continuous sign language recognition using CNN+LSTM/3D CNN/GCN/Encoder-Decoder

## Requirements

- Download and extract **[CSL Dataset](http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html)**
- Download and install **[PyTorch](https://pytorch.org/)**

## Isolated Sign Language Recognition

### CNN+LSTM

1. **four layers of Conv2d + one layer of LSTM**

   | Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
   | ------------ | ------- | ------- | ------------- | -------------- |
   | CSL_Isolated | 100     | 25,000  | 82.08%        | 0.734426       |
   | CSL_Isolated | 500     | 125,000 | 71.71%        | 1.332122       |

2. **ResNet + one layer of LSTM**

   | Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
   | ------------ | ------- | ------- | ------------- | -------------- |
   | CSL_Isolated | 100     | 25,000  | 93.54%        | 0.245582       |
   | CSL_Isolated | 500     | 125,000 | 83.17%        | 0.748759       |

### 3D CNN

1. **three layers of Conv3d**

   | Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
   | ------------ | ------- | ------- | ------------- | -------------- |
   | CSL_Isolated | 100     | 25,000  | 58.86%        | 1.560049       |
   | CSL_Isolated | 500     | 125,000 | 45.07%        | 2.255563       |
   
2. **3D ResNet**

   | Method    | Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
   | --------- | ------------ | ------- | ------- | ------------- | -------------- |
   | ResNet18  | CSL_Isolated | 100     | 25,000  | 93.30%        | 0.246169       |
   | ResNet18  | CSL_Isolated | 500     | 125,000 | 79.42%        | 0.800490       |
   | ResNet34  | CSL_Isolated | 100     | 25,000  | 94.78%        | 0.207592       |
   | ResNet34  | CSL_Isolated | 500     | 125,000 | 81.61%        | 0.750424       |
   | ResNet50  | CSL_Isolated | 100     | 25,000  | 94.36%        | 0.232631       |
   | ResNet50  | CSL_Isolated | 500     | 125,000 | 83.15%        | 0.803212       |
   | ResNet101 | CSL_Isolated | 100     | 25,000  | 95.26%        | 0.205430       |
   | ResNet101 | CSL_Isolated | 500     | 125,000 | 83.18%        | 0.751727       |

3. **ResNet (2+1)D**

   | Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
   | ------------ | ------- | ------- | ------------- | -------------- |
   | CSL_Isolated | 100     | 25,000  | 98.68%        | 0.043099       |
   | CSL_Isolated | 500     | 125,000 | 94.85%        | 0.234880       |

### GCN

| Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
| ------------ | ------- | ------- | ------------- | -------------- |
| CSL_Skeleton | 100     | 25,000  | 79.20%        | 0.737053       |
| CSL_Skeleton | 500     | 125,000 | 66.64%        | 1.165872       |

### Skeleton+LSTM

| Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
| ------------ | ------- | ------- | ------------- | -------------- |
| CSL_Skeleton | 100     | 25,000  | 84.30%        | 0.488253       |
| CSL_Skeleton | 500     | 125,000 | 70.62%        | 1.078730       |

## Continuous Sign Language Recognition

### Encoder-Decoder

*Encoder is ResNet18+LSTM, and Decoder is LSTM*

| Dataset             | Sentences | Samples | Best Test Wer | Best Test Loss |
| ------------------- | --------- | ------- | ------------- | -------------- |
| CSL_Continuous      | 100       | 25,000  | 1.01%         | 0.034636       |
| CSL_Continuous_Char | 100       | 25,000  | 1.19%         | 0.049449       |

## References

- [Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?](https://arxiv.org/pdf/1711.09577.pdf)

- [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/pdf/1801.07455.pdf)
- [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248)
- [SIGN LANGUAGE RECOGNITION WITH LONG SHORT-TERM MEMORY](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7532884)
- https://github.com/HHTseng/video-classification
- https://github.com/kenshohara/3D-ResNets-PyTorch

- https://github.com/bentrevett/pytorch-seq2seq

