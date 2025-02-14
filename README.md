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
|         | 0.8956 | 0.8318   | 0.6463    | 0.8689  | 0.7413   |

### Seed 2: w/o pos_weight
| Metric  | AUROC  | Accuracy | Precision | Recall  | F1 Score |
|---------|--------|----------|-----------|---------|----------|
|         | 0.8725 | 0.8364   | 0.6897    | 0.6897  | 0.6897   |

### Seed 2: w/ pos_weight

| Metric  | AUROC  | Accuracy | Precision | Recall  | F1 Score |
|---------|--------|----------|-----------|---------|----------|
|         | 0.8669 | 0.7636   | 0.5385    | 0.7241  | 0.6176   |

## VGG19
### w/o pos_weight
| Metric  | AUROC  | Accuracy | Precision | Recall  | F1 Score |
|---------|--------|----------|-----------|---------|----------|
|         | 0.5983 | 0.7273   | 0.4667    | 0.2414  | 0.3182   |

### w/ pos_weight
| Metric  | AUROC  | Accuracy | Precision | Recall  | F1 Score |
|---------|--------|----------|-----------|---------|----------|
|         | 0.6972 | 0.7864   | 0.7391    | 0.2931  | 0.4198   |