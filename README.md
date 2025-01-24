[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/cZnpr7Ns)
# E4040 2024 Fall Project
## DenseNet-BC Reproduction on CIFAR-10 and CIFAR-100

Overview
This project reproduces the DenseNet-BC architecture and applies it to the CIFAR-10 and CIFAR-100 datasets to evaluate its performance. The goal is to validate the original findings and explore the impact of the Bottleneck and Compression techniques on model accuracy. The project includes training a DenseNet model, implementing key architectural features, and addressing challenges related to computational resources.

Key Features
DenseNet-BC Architecture: Reproduced with bottleneck and compression layers for efficient feature reuse and reduced parameters.
Datasets: Evaluated on CIFAR-10 (10 classes) and CIFAR-100 (100 classes).
Training: Utilizes Stochastic Gradient Descent (SGD) with momentum and sparse categorical cross-entropy loss.
Regularization: Implements dropout and learning rate scheduling.
Computational Constraints: Acknowledges the impact of limited computational resources on model performance and training time.
Results
Error Rate: Higher than expected compared to the original paper—about 5 times higher for CIFAR-10 and 3 times higher for CIFAR-100.
Bottleneck and Compression: No significant improvement observed for the tested parameter scale, although models with Bottleneck and Compression generally outperformed those without.
Hardware Limitations: Computational resources (GPU memory) were a limiting factor, leading to a smaller scale of experimentation compared to the original paper.
Usage
Load Data: The load_data.py script handles loading and preprocessing the CIFAR-10 and CIFAR-100 datasets. You can modify this script to load different datasets if needed.

Model Training: To train the DenseNet model, run the main.ipynb notebook. It will automatically load the dataset, build the model, and train it with the specified hyperparameters.

Evaluation: Training results, including loss and accuracy, will be printed at each epoch. You can also visualize the model's performance using Matplotlib.

Hyperparameter Tuning: Adjust the batch size, learning rate, and epochs in the notebook to experiment with different configurations.

Challenges and Future Work
Computational Resources: Due to hardware limitations, we could not explore larger models with more parameters or run extensive hyperparameter searches.
Overfitting: Regularization techniques such as dropout were necessary to mitigate overfitting during training.
Future Work: Future experiments can explore deeper architectures, larger datasets like ImageNet, and more sophisticated optimization techniques like AdamW.
Results and Insights
Bottleneck and Compression: We did not observe significant improvements with Bottleneck and Compression layers, potentially due to the scale of our model.
Hardware Constraints: Limited GPU memory (7.5 GB) restricted our ability to scale the model for larger parameter configurations.
Conclusion
This project successfully reproduced the core DenseNet-BC architecture and evaluated it on CIFAR-10 and CIFAR-100 datasets. While the results showed trends consistent with the original paper, such as improved accuracy with Bottleneck and Compression, the error rates were higher due to computational constraints. The project highlighted the importance of parameter tuning, regularization, and sufficient hardware for training deep neural networks.

License
This project is licensed under the MIT License - see the LICENSE file for details.

# Organization of this directory
To be populated by students, as shown in previous assignments.

TODO: Create a directory/file tree
```
├── .DS_Store
├── .git
│   ├── COMMIT_EDITMSG
│   ├── FETCH_HEAD
│   ├── HEAD
│   ├── ORIG_HEAD
│   ├── branches
│   ├── config
│   ├── description
│   ├── hooks
│   │   ├── applypatch-msg.sample
│   │   ├── commit-msg.sample
│   │   ├── fsmonitor-watchman.sample
│   │   ├── post-update.sample
│   │   ├── pre-applypatch.sample
│   │   ├── pre-commit.sample
│   │   ├── pre-push.sample
│   │   ├── pre-rebase.sample
│   │   ├── pre-receive.sample
│   │   ├── prepare-commit-msg.sample
│   │   └── update.sample
│   ├── index
│   ├── info
│   │   └── exclude
│   ├── logs
...
├── densenet.py
├── load_data.py
├── main.ipynb
└── train.py
```
