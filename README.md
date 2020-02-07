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

2. **resnet152 + one layer of LSTM**

   three channels(rgb):

   | Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
   | ------------ | ------- | ------- | ------------- | -------------- |
   | CSL_Isolated | 100     | 25,000  |               |                |
   | CSL_Isolated | 500     | 125,000 |               |                |

### 3D CNN

1. **three layers of Conv3d**

   one channel(grayscale):

   | Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
   | ------------ | ------- | ------- | ------------- | -------------- |
   | CSL_Isolated | 100     | 25,000  | 97.64%        | 0.094761       |
   | CSL_Isolated | 500     | 125,000 | 92.24%        | 0.295053       |



### GCN



## Todos

- 3D Resnet

- I3D