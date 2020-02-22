# SLR
sign language recognition using CNN+LSTM, 3D CNN, GCN and their variants

## Models

### CNN+LSTM

### 3D CNN

1. **three layers of Conv3d**

   | Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
   | ------------ | ------- | ------- | ------------- | -------------- |
   | CSL_Skeleton | 100     | 25,000  | 58.86%        | 1.560049       |
   
2. **3D ResNet**

   | Method   | Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
   | -------- | ------------ | ------- | ------- | ------------- | -------------- |
   | Resnet18 | CSL_Isolated | 100     | 25,000  | 84.20%        | 0.733129       |

### GCN

| Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
| ------------ | ------- | ------- | ------------- | -------------- |
| CSL_Skeleton | 100     | 25,000  | 79.20%        | 0.737053       |
| CSL_Skeleton | 500     | 125,000 | 66.64%        | 1.165872       |

### Skeleton+LSTM

| Dataset      | Classes | Samples | Best Test Acc | Best Test Loss |
| ------------ | ------- | ------- | ------------- | -------------- |
| CSL_Skeleton | 100     | 25,000  | 84.36%        | 0.495949       |
| CSL_Skeleton | 500     | 125,000 | 70.62%        | 1.078730       |

## Todos

- validation
- load pretrained model
- confusion matrix/visualize error output
- seq-to-seq learning
- attension machanism
- CTC loss

