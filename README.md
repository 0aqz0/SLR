# SLR
sign language recognition using CNN+LSTM, 3D CNN, GCN and their variants

## Requirements

- Download and extract **[CSL Dataset](http://home.ustc.edu.cn/~pjh/openresources/cslr-dataset-2015/index.html)**
- Download and install **[PyTorch](https://pytorch.org/)**

## Models

### CNN+LSTM

1. **four layers of Conv2d + one layer of LSTM**

   | Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
   | ------------ | ------- | ------- | ------------- | -------------- |
   | CSL_Isolated | 100     | 25,000  | 82.08%        | 0.734426       |
   | CSL_Isolated | 500     | 125,000 | 71.71%        | 1.332122       |

2. **ResNet + one layer of LSTM**

   | Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
   | ------------ | ------- | ------- | ------------- | -------------- |
   | CSL_Isolated | 100     | 25,000  | 92.06%        | 0.279240       |
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
   | CSL_Isolated | 100     | 25,000  | 0.043099      | 98.68%         |
   | CSL_Isolated | 500     | 125,000 | 0.234880      | 94.85%         |

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

## Todos

- seq-to-seq learning
- CTC loss

