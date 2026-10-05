# multilayer_perceptron: Neural Network from Scratch 🧠

[![42 School](https://img.shields.io/badge/42-School-blue)](https://42firenze.it/)
[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-From%20Scratch-013243?logo=numpy)](https://numpy.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-MLP-orange)](https://en.wikipedia.org/wiki/Multilayer_perceptron)
[![Dataset](https://img.shields.io/badge/Dataset-Breast%20Cancer%20Wisconsin-brightgreen)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
[![GitHub license](https://img.shields.io/github/license/Nazar963/42_multilayer_perceptron)](https://github.com/Nazar963/42_multilayer_perceptron/blob/master/LICENSE)

A complete implementation of a **Multilayer Perceptron (MLP) from scratch in Python**, built without machine-learning libraries.

The network is trained to classify breast tumors as **benign (B)** or **malignant (M)** using the Breast Cancer Wisconsin Diagnostic dataset.

The project implements the complete neural-network pipeline manually: dataset preprocessing, weight initialization, feedforward propagation, loss computation, backpropagation, gradient descent, model persistence and prediction.

## Table of Contents 📖

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Neural Network Architecture](#neural-network-architecture)
- [Mathematical Foundation](#mathematical-foundation)
- [Implementation](#implementation)
- [Training Pipeline](#training-pipeline)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Persistence](#model-persistence)
- [Bonus Features](#bonus-features)
- [Limitations](#limitations)
- [Resources](#resources)
- [License](#license)

## Project Overview

`multilayer_perceptron` is a 42 School project focused on understanding how neural networks actually work internally.

Instead of relying on frameworks such as TensorFlow, PyTorch or Scikit-learn, the core machine-learning algorithms are implemented manually using **NumPy**.

The project covers:

- Dataset exploration and visualization
- Dataset shuffling and splitting
- Feature standardization
- Configurable neural-network topology
- He Uniform weight initialization
- Feedforward propagation
- ReLU activation
- Softmax output activation
- Binary Cross-Entropy loss
- Backpropagation
- Gradient descent
- Training and validation metrics
- Model serialization
- Prediction using a previously trained model
- Early stopping
- Learning-curve visualization

**Project Requirements:**

- Implement the neural network without ML frameworks
- Use at least **two hidden layers**
- Implement feedforward propagation manually
- Implement backpropagation manually
- Implement gradient descent manually
- Use a validation dataset during training
- Display training and validation metrics
- Save the trained model
- Load the trained model from a separate prediction program

NumPy, Pandas and Matplotlib are used for numerical operations, dataset manipulation and visualization, but **not to provide pre-built neural-network algorithms**.

---

## Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic Dataset**.

The original dataset contains:

| Property | Value |
|----------|------:|
| Samples | 569 |
| Input features | 30 |
| Classes | 2 |
| Benign samples | 357 |
| Malignant samples | 212 |

Each row represents measurements extracted from a digitized image of a breast mass.

The target column contains:

```text
B → Benign
M → Malignant
```

Internally, the labels are encoded as:

```text
B → 0
M → 1
```

After removing the ID column, each sample contains **30 numerical features**.

### Train / Validation Split

The dataset is shuffled reproducibly using:

```python
data.sample(frac=1, random_state=51).reset_index(drop=True)
```

and divided approximately as:

```text
Training set:   455 samples × 30 features
Validation set: 114 samples × 30 features
```

This corresponds to an approximately **80 / 20 split**.

---

## Neural Network Architecture

The default network architecture is:

```text
30 Inputs
    │
    ▼
┌───────────────┐
│ Hidden Layer  │ 24 neurons
│     ReLU      │
└───────────────┘
    │
    ▼
┌───────────────┐
│ Hidden Layer  │ 24 neurons
│     ReLU      │
└───────────────┘
    │
    ▼
┌───────────────┐
│ Output Layer  │ 2 neurons
│    Softmax    │
└───────────────┘
    │
    ▼
Benign / Malignant
```

Therefore, the default topology is:

```text
[30, 24, 24, 2]
```

The hidden-layer configuration is not hardcoded and can be changed at training time.

For example:

```text
42,10,20
```

produces:

```text
[30, 42, 10, 20, 2]
```

with weight matrices:

```text
(30, 42)
(42, 10)
(10, 20)
(20, 2)
```

---

## Mathematical Foundation

### Feature Standardization

Features can have very different numerical ranges.

Each feature is standardized using the mean and standard deviation calculated **only from the training set**:

$$
x' = \frac{x - \mu}{\sigma}
$$

where:

- $x$ = original feature
- $\mu$ = training-set mean
- $\sigma$ = training-set standard deviation

The same $\mu$ and $\sigma$ are then applied to the validation data.

This prevents information from the validation set from leaking into training.

---

### Weight Initialization

Weights are initialized using a **He-style uniform initialization**.

For a layer with $n$ input neurons:

$$
limit = \sqrt{\frac{6}{n}}
$$

Weights are sampled from:

$$
W \sim U(-limit, limit)
$$

Biases are initialized to zero.

A fixed NumPy seed is used for reproducibility:

```python
np.random.seed(51)
```

---

### Feedforward Propagation

For every layer, the weighted sum is:

$$
Z^{[l]} = A^{[l-1]}W^{[l]} + b^{[l]}
$$

The activation function is then applied:

$$
A^{[l]} = f(Z^{[l]})
$$

The information therefore flows through the network as:

```text
Input
  ↓
Weighted Sum
  ↓
Activation
  ↓
Hidden Layer
  ↓
...
  ↓
Softmax
  ↓
Prediction
```

---

### ReLU

Hidden layers use the **Rectified Linear Unit**:

$$
ReLU(z) = \max(0,z)
$$

Its derivative is:

$$
ReLU'(z) =
\begin{cases}
1 & z > 0 \\
0 & z \leq 0
\end{cases}
$$

Python implementation:

```python
def ReLU(z):
    return np.maximum(0, z)


def ReLU_derivative(z):
    return (z > 0).astype(float)
```

---

### Softmax

The output layer contains two neurons and uses Softmax to convert its values into probabilities:

$$
Softmax(z_i) =
\frac{e^{z_i}}
{\sum_j e^{z_j}}
$$

For example:

```text
[2.4, 0.7]
        ↓
Softmax
        ↓
[0.845, 0.155]
```

which can be interpreted as approximately:

```text
84.5% → Benign
15.5% → Malignant
```

For numerical stability, the maximum value is subtracted before calculating the exponential.

---

### Binary Cross-Entropy

The network's error is measured using Binary Cross-Entropy:

$$
L =
-\frac{1}{N}
\sum_{n=1}^{N}
\left[
y_n\log(p_n)
+
(1-y_n)\log(1-p_n)
\right]
$$

Probabilities are clipped to avoid undefined operations such as:

```text
log(0)
```

Example:

```python
e = 1e-15
p = np.clip(Y_pred, e, 1 - e)
```

---

### Backpropagation

Backpropagation calculates how much each weight and bias contributed to the final error.

With Softmax and Cross-Entropy, the output-layer error simplifies to:

$$
dZ = A - Y
$$

The gradients are then calculated backward through the network:

$$
dW^{[l]} =
\frac{1}{N}
(A^{[l-1]})^T dZ^{[l]}
$$

$$
db^{[l]} =
\frac{1}{N}
\sum dZ^{[l]}
$$

For hidden layers:

$$
dA^{[l-1]} =
dZ^{[l]}(W^{[l]})^T
$$

and:

$$
dZ^{[l-1]} =
dA^{[l-1]} \odot ReLU'(Z^{[l-1]})
$$

Backpropagation therefore moves from the output layer toward the input layer.

---

### Gradient Descent

Once the gradients have been calculated, the parameters are updated using:

$$
W = W - \eta dW
$$

$$
b = b - \eta db
$$

where $\eta$ is the **learning rate**.

Example:

```python
W[i] = W[i] - n * dW[i]
B[i] = B[i] - n * dB[i]
```

This process is repeated for every epoch.

---

## Implementation

The project is divided into three main programs.

### `separazione.py`

Responsible for dataset preparation:

- Loads `data.csv`
- Removes unnecessary columns
- Converts diagnosis labels
- Shuffles the dataset
- Splits training and validation data
- Calculates training mean and standard deviation
- Standardizes numerical features
- Generates the datasets used during training

---

### `addestramento.py`

Contains the actual neural-network implementation.

Main responsibilities:

- Read hyperparameters
- Build the requested topology
- Initialize weights and biases
- Perform feedforward propagation
- Calculate the loss
- Calculate accuracy
- Perform backpropagation
- Apply gradient descent
- Track training metrics
- Track validation metrics
- Perform early stopping when enabled
- Save the final model

Conceptually:

```python
for epoch in range(epochs):

    # Feedforward
    Z, A = feedForward(X_train, W, B)

    # Training metrics
    bce = loss(A[-1], Y_train)
	P_train = np.argmax(A[-1], axis=1)
	Ac_train = np.mean(P_train == Y_train)

    # Backpropagation
    DW, DB = backPropagation(X_train, Y_train, W, Z, A)

    # Gradient descent
    W, B = gradient_descent(W, B, DW, DB, n)

    # Validation
    _, A_validate = feedForward(X_validate, W, B)

    # Validation metrics
    bce_validate = loss(A_validate[-1], Y_validate)
    P_validate = np.argmax(A_validate[-1], axis=1)
	Ac_validate = np.mean(P_validate == Y_validate)
```

---

### `previsione.py`

Loads the previously trained model and uses it to classify unseen validation data.

The prediction program:

- Loads `model.npz`
- Restores the network topology
- Restores weights and biases
- Loads the validation data
- Performs feedforward propagation
- Calculates predictions
- Calculates final loss
- Calculates final accuracy

Because the architecture is stored with the model, the prediction program can dynamically reconstruct networks with different hidden-layer configurations.

---

## Training Pipeline

The complete workflow is:

```text
                data.csv
                    │
                    ▼
             Dataset Shuffle
                    │
                    ▼
              Train / Valid
                  Split
                    │
                    ▼
             Standardization
                    │
                    ▼
        ┌───────────────────────┐
        │ Weight Initialization │
        └───────────────────────┘
                    │
                    ▼
               Feedforward
                    │
                    ▼
                Prediction
                    │
                    ▼
              Loss + Accuracy
                    │
                    ▼
             Backpropagation
                    │
                    ▼
            Gradient Descent
                    │
                    │
              Repeat Epochs
                    │
                    ▼
            Validation Metrics
                    │
                    ▼
                model.npz
                    │
                    ▼
              previsione.py
```

---

## Features

### Core Implementation

| Feature | Description |
|---------|-------------|
| **Dataset preprocessing** | Clean and prepare the Breast Cancer dataset |
| **Train/validation split** | Reproducible dataset separation |
| **Z-score standardization** | Normalize features using training statistics |
| **Dynamic architecture** | Configure hidden layers and neuron counts |
| **He initialization** | Weight initialization adapted for ReLU |
| **Feedforward** | Fully implemented matrix-based propagation |
| **ReLU** | Hidden-layer activation function |
| **Softmax** | Two-neuron probability output |
| **Binary Cross-Entropy** | Training loss function |
| **Backpropagation** | Manual gradient calculation |
| **Gradient descent** | Manual parameter optimization |
| **Accuracy tracking** | Training and validation accuracy |
| **Model persistence** | Save and reload network parameters |
| **Dynamic prediction** | Prediction program reconstructs saved topology |

### Core Techniques

- **Matrix multiplication** for dense neural-network layers
- **Vectorized operations** using NumPy
- **One-hot encoding** for output labels
- **Argmax classification** for final predictions
- **Numerically stable Softmax**
- **Probability clipping** for Cross-Entropy
- **Reverse layer traversal** during backpropagation
- **Reproducible initialization** using fixed random seeds

---

## Installation

### Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib

No TensorFlow, PyTorch, Keras or Scikit-learn neural-network implementation is used.

### Setup

```bash
# Clone repository
git clone https://github.com/Nazar963/42_multilayer_perceptron.git

cd 42_multilayer_perceptron

# Optional: create virtual environment
python3 -m venv .venv

source .venv/bin/activate

# Install dependencies
pip install numpy pandas matplotlib
```

---

## Usage

### 1. Prepare the Dataset

Place the Breast Cancer Wisconsin dataset in the repository as:

```text
data.csv
```

Then run:

```bash
python3 separazione.py
```

This prepares the training and validation datasets.

---

### 2. Train the Network

Run:

```bash
python3 addestramento.py
```

Example configuration:

```text
Learning rate: 0.01
Epochs: 1000
Hidden layers: 24,24
```

This creates the topology:

```text
30 → 24 → 24 → 2
```

The program displays training and validation metrics during the learning process.

---

### 3. Run Prediction

After training:

```bash
python3 previsione.py
```

Example output:

```text
(30, 24) (24,)
(24, 24) (24,)
(24, 2) (2,)
(114, 30) (114,)
Questa è la topologia della rete: [30 24 24  2]
--------
Esempi: 114
Loss: 0.1041
Accuracy: 0.9561
```

The exact values can change depending on hyperparameters and network architecture.

---

### Different Architecture

The topology can be modified without rewriting the neural-network code.

For example:

```text
Learning rate: 0.1
Epochs: 1000
Hidden layers: 42,10,20
```

creates:

```text
30 → 42 → 10 → 20 → 2
```

instead of the default:

```text
30 → 24 → 24 → 2
```

---

## Model Persistence

After training, network parameters are saved inside:

```text
model.npz
```

The model contains the information required to reconstruct the network, including:

- Weight matrices
- Bias vectors
- Network topology

Conceptually:

```python
model_data = {}
for i in range(len(W)):
	model_data[f"w{i + 1}"] = W[i]
	model_data[f"b{i + 1}"] = B[i]
model_data["layers"] = layers
print("saving './model model.npz' to disk...")
np.savez("model.npz", **model_data)
```

The prediction program can therefore load a model without having its hidden-layer architecture hardcoded.

---

## Bonus Features

Several additional features were implemented beyond the mandatory requirements.

### Early Stopping

Training can automatically stop when the validation loss no longer improves.

A patience mechanism prevents stopping after a single bad epoch.

Conceptually:

```text
Validation improves
        ↓
Save best model
        ↓
No improvement
        ↓
Increase patience counter
        ↓
Counter reaches limit
        ↓
Stop training
        ↓
Restore best weights
```

This prevents later epochs from replacing a model that performed better on validation data.

---

### Training History

Metrics are recorded for every epoch:

```text
Training Loss
Validation Loss
Training Accuracy
Validation Accuracy
```

These values can be stored and later visualized.

---

### Learning Curves

Multiple network configurations can be trained and compared by plotting their learning histories.

This makes it possible to analyze the effect of parameters such as:

- Learning rate
- Number of epochs
- Number of hidden layers
- Number of neurons per layer

Example comparison:

```text
Configuration A
Learning rate: 0.01
Hidden layers: 24,24

Configuration B
Learning rate: 0.1
Hidden layers: 42,10,20
```

The resulting loss and accuracy curves can then be compared visually.

---

## Limitations

This implementation intentionally remains relatively simple because its purpose is to understand the foundations of neural networks.

1. **Full-batch Gradient Descent** — the complete training dataset is processed at once.
2. **No advanced optimizers** — Adam, RMSProp and Momentum are not implemented.
3. **No GPU acceleration** — computation is performed with NumPy on the CPU.
4. **Binary classification** — the current project targets benign/malignant classification.
5. **No convolutional layers** — only fully connected dense layers are implemented.
6. **Limited regularization** — techniques such as dropout and L1/L2 regularization are not part of the core implementation.
7. **Small dataset** — the Breast Cancer Wisconsin dataset contains only 569 samples.

These limitations are intentional: the objective is to implement and understand the fundamental algorithms rather than compete with production ML frameworks.

---

## Resources

1. [Multilayer Perceptron — Wikipedia](https://en.wikipedia.org/wiki/Multilayer_perceptron) — MLP fundamentals
2. [Backpropagation — Wikipedia](https://en.wikipedia.org/wiki/Backpropagation) — Gradient calculation
3. [Gradient Descent — Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent) — Optimization algorithm
4. [Rectifier — Wikipedia](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) — ReLU activation
5. [Softmax Function — Wikipedia](https://en.wikipedia.org/wiki/Softmax_function) — Output probabilities
6. [Cross-Entropy — Wikipedia](https://en.wikipedia.org/wiki/Cross-entropy) — Loss function
7. [Breast Cancer Wisconsin Dataset — UCI](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) — Dataset reference
8. [NumPy Documentation](https://numpy.org/doc/stable/) — Numerical computation

---

## 🤝 Contributing

Feel free to submit issues or pull requests if you have suggestions for improving the implementation or adding new features.


## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.


## 📧 Contact

For questions or feedback, please open an issue in the repository.


## ⭐ Star this repository if you found it helpful!

[![GitHub stars](https://img.shields.io/github/stars/Nazar963/42_multilayer_perceptron?style=social)](https://github.com/Nazar963/42_multilayer_perceptron/stargazers)

---
🧠 *"What I cannot create, I do not understand."* — Richard Feynman  
[![42 School](https://img.shields.io/badge/42-profile-blue)](https://profile-v3.intra.42.fr/users/naal-jen)
[![GitHub Profile](https://img.shields.io/badge/GitHub-Nazar963-lightgrey)](https://github.com/Nazar963)
[![GitHub Follow](https://img.shields.io/github/followers/Nazar963?style=social)](https://github.com/Nazar963)


## 🍀 Good luck

Good luck with your `multilayer_perceptron` project at 42! 🚀