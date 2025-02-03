# Chest X Ray Classification
Under construction

# Run
configure basic parameters in settings.yml
```sh
python train.py
```

# Test Results
DenseNet121
| Metric      | AUROC  | Accuracy | Precision | Recall  | F1 Score |
|-------------|--------|----------|-----------|---------|----------|
|             | 0.8859 | 0.8409   | 0.8043    | 0.5873  | 0.6789   |

VGG11
| Metric      | AUROC  | Accuracy | Precision | Recall  | F1 Score |
|-------------|--------|----------|-----------|---------|----------|
|             | 0.5676 | 0.7364   | 1.0000    | 0.0794  | 0.1471   |