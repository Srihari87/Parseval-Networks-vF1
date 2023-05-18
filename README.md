---
license: gpl-3.0
language:
- en
metrics:
- accuracy
- recall
- precision
- roc_auc
pipeline_tag: image-classification
---
## Adversarial Examples for improving the robustness of Eye-State Classification 👁 👁 :

### First Aim:
Project aims to improve the robustness of the model by adding the adversarial examples to the training dataset.
We investigated that the robustness of the models on the clean test data are always better than the attacks even though added the pertubated data to the training data.
### Second Aim:

Using adversarial examples, the project aims to improve the robustness and accuracy of a machine learning model which detects the eye-states against small perturbation of an image and to solve the misclassification problem caused by natural transformation.
### Methodologies

* Develop Wide Residual Network and Parseval Network.
* Train Neural Networks using training dataset.
* Construct the AEs using FGSM and Random Noise.
#### The approach for the first aim.
===================================================================
* Train Neural Networks by adding Adversarial Examples (AEs) to the training dataset.
* Evaluate the models on the original test dataset.

#### The approach for the second aim.
===================================================================
* Train Neural Networks using Adversarial Training with AEs.
* Attack the new model with different perturbated test dataset.

### Neural Network Models

#### Wide Residual Network

* Baseline of the Model

#### Parseval Network

* [Orthogonality Constraint in Convolutional Layers](https://huggingface.co/Sefika/parseval-network/blob/main/models/Parseval_Networks/constraint.py)
* [Convexity Constraint in Aggregation Layers](https://huggingface.co/Sefika/parseval-network/blob/main/models/Parseval_Networks/convexity_constraint.py)

#### Convolutional Neural Network

#### Adversarial Examples

##### Fast Gradient Sign Method
[Examples](https://huggingface.co/Sefika/parseval-network/blob/main/visualization/Adversarial_Images.ipynb)

### Evaluation

* To evaluate the result of the neural network, Signal to Noise Ratio (SNR) is used as metric.
* Use transferability of AEs to evaluate the models.

## Development 

#### Models:

``` bash

adversarial_examples_parseval_net/models
├── FullyConectedModels
│   ├── model.py
│   └── parseval.py
├── Parseval_Networks
│   ├── constraint.py
│   ├── convexity_constraint.py
│   ├── parsevalnet.py
├── _utility.py
└── wideresnet
    └── wresnet.py


```

### Final Results:

* [The results of the first approach with FGSM](https://huggingface.co/Sefika/parseval-network/tree/main/logs/AEModels)
* [The results of the first approach with Random Noise](https://huggingface.co/Sefika/parseval-network/tree/main/logs/RandomNoisemodels)
* [The results of the second approach](https://huggingface.co/Sefika/parseval-network/tree/main/logs)


References
============
[1] Cisse, Bojanowski, Grave, Dauphin and Usunier, Parseval Networks: Improving Robustness to Adversarial Examples, 2017.

[2] Zagoruyko and Komodakis, Wide Residual Networks, 2016.

``` 

@misc{ParsevalNetworks,
  author= "Moustapha Cisse, Piotr Bojanowski, Edouard Grave, Yann Dauphin, Nicolas Usunier"
  title="Parseval Networks: Improving Robustness to Adversarial Examples"
  year= "2017"
}
```

``` 

@misc{Wide Residual Networks
  author= "Sergey Zagoruyko, Nikos Komodakis"
  title= "Wide Residual Networks"
  year= "2016"
}
```

### Author

Sefika Efeoglu

Research Project, Data Science MSc, University of Potsdam