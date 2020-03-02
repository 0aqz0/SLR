# SLR
sign language recognition using CNN+LSTM, 3D CNN, GCN and their variants

## Models

### CNN+LSTM

### 3D CNN

1. **three layers of Conv3d**

   | Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
   | ------------ | ------- | ------- | ------------- | -------------- |
   | CSL_Isolated | 100     | 25,000  | 58.86%        | 1.560049       |
   | CSL_Isolated | 500     | 125,000 |               |                |
   
2. **3D ResNet**

   | Method   | Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
   | -------- | ------------ | ------- | ------- | ------------- | -------------- |
   | ResNet18 | CSL_Isolated | 100     | 25,000  | 93.30%        | 0.246169       |
   | ResNet18 | CSL_Isolated | 500     | 125,000 | 79.42%        | 0.800490       |
   | ResNet34 | CSL_Isolated | 100     | 25,000  | 94.78%        | 0.207592       |
   | ResNet34 | CSL_Isolated | 500     | 125,000 | 81.61%        | 0.750424       |
   | ResNet50 | CSL_Isolated | 100     | 25,000  | 94.36%        | 0.232631       |
   | ResNet50 | CSL_Isolated | 500     | 125,000 |               |                |

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

- r2+1d, slowfast
- seq-to-seq learning
- attension machanism
- CTC loss

