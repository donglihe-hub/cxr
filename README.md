# Chest X Ray Classification
Under construction

# Run
configure basic parameters in settings.yml
```sh
python train.py
```

# Test Results
## DenseNet121 
### Seed 1: w/o pos_weight
| Metric  | AUROC  | Accuracy | Precision | Recall  | F1 Score |
|---------|--------|----------|-----------|---------|----------|
|         | 0.8777 | 0.8227   | 0.7115    | 0.6066  | 0.6549   |

### Seed 1: w/ pos_weight
| Metric  | AUROC  | Accuracy | Precision | Recall  | F1 Score |
|---------|--------|----------|-----------|---------|----------|
|         | 0.8745 | 0.8364   | 0.6964    | 0.6724  | 0.6842   |

## VGG19_bn
### w/o pos_weight
| Metric  | AUROC  | Accuracy | Precision | Recall  | F1 Score |
|---------|--------|----------|-----------|---------|----------|
|         | 0.6296 | 0.7500   | 0.5405    | 0.3448  | 0.4211   |

### w/ pos_weight
| Metric   | AUROC  | Accuracy | Precision | Recall | F1 Score |
|----------|--------|----------|-----------|--------|----------|
|          | 0.5821 | 0.2636   | 0.2636    | 1.0000 | 0.4173   |

## ResNet50
### w/o pos_weight
| Metric    | AUROC  | Accuracy | Precision | Recall  | F1 Score |
|-----------|--------|----------|-----------|---------|----------|
|           | 0.9043 | 0.8409   | 0.7091    | 0.6724  | 0.6903   |

### w/ pos_weight
| Metric    | AUROC  | Accuracy | Precision | Recall  | F1 Score |
|-----------|--------|----------|-----------|---------|----------|
|           | 0.8631 | 0.8136   | 0.6735    | 0.5690  | 0.6168   |

## 3-layer CNN
### w/o pos_weight
| Metric    | AUROC  | Accuracy | Precision | Recall  | F1 Score |
|-----------|--------|----------|-----------|---------|----------|
|           | 0.6853 | 0.7864   | 0.7895    | 0.2586  | 0.3896   |

### w/ pos_weight
| Metric    | AUROC  | Accuracy | Precision | Recall | F1 Score |
|-----------|--------|----------|-----------|--------|----------|
|           | 0.6672 | 0.2636   | 0.2636    | 1.0000 | 0.4173   |

## EfficintNetB7
### w/o pos_weight
| Metric    | AUROC  | Accuracy | Precision | Recall  | F1 Score |
|-----------|--------|----------|-----------|---------|----------|
|           | 0.8992 | 0.8136   | 0.6133    | 0.7931  | 0.6917   |

### w/ pos_weight
| Metric    | AUROC  | Accuracy | Precision | Recall  | F1 Score |
|-----------|--------|----------|-----------|---------|----------|
|           | 0.9215 | 0.8591   | 0.7368    | 0.7241  | 0.7304   |
