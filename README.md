# SLR
sign language recognition using CNN+LSTM, 3D CNN, GCN and their variants

## Models

### CNN+LSTM

1. **four layers of Conv2d + one layer of LSTM**

   one channel(grayscale):

   | Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
   | ------------ | ------- | ------- | ------------- | -------------- |
   | CSL_Isolated | 100     | 25,000  | 94.88%        | 0.215799       |
   | CSL_Isolated | 500     | 125,000 | 91.09%        | 0.412514       |

2. **ResNet + one layer of LSTM**

   three channels(rgb):

   | Model     | Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
   | --------- | ------------ | ------- | ------- | ------------- | -------------- |
   | ResNet152 | CSL_Isolated | 100     | 25,000  | 93.80%        | 0.209352       |
   | ResNet152 | CSL_Isolated | 500     | 125,000 |               |                |

### 3D CNN

1. **three layers of Conv3d**

   one channel(grayscale):

   | Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
   | ------------ | ------- | ------- | ------------- | -------------- |
   | CSL_Isolated | 100     | 25,000  | 97.64%        | 0.094761       |
   | CSL_Isolated | 500     | 125,000 | 92.24%        | 0.295053       |

2. **3D ResNet**

   three channels(rgb):

   | Model    | Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
   | -------- | ------------ | ------- | ------- | ------------- | -------------- |
   | Resnet18 | CSL_Isolated | 100     | 25,000  | 98.72%        | 0.051288       |
   | ResNet18 | CSL_Isolated | 500     | 125,000 |               |                |

### GCN



## Todos

- GCN
- 3D Resnet from torchvision
- I3D
- seq-to-seq learning
- attension machanism
- CTC loss

