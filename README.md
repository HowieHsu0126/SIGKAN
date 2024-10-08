# SIGKAN: Sociologically-Informed Graph Kolmogorov-Arnold Networks for Social Influence Prediction

## Overview
SIGKAN (Sociologically-Informed Graph Kolmogorov-Arnold Networks) is a deep learning framework for predicting social influence on graphs, leveraging sociological theories and advanced neural network techniques. The key idea is to integrate the Bounded Confidence Model (BCM) for influence propagation and Kolmogorov-Arnold Networks (KANs) for feature updates, enhancing model interpretability and expressiveness. SIGKAN is designed to address the limitations of traditional opinion dynamics models and current deep learning methods on graphs by combining theoretical social influence models with state-of-the-art graph-based learning.

### Abstract
Social influence plays a crucial role in both offline interactions and virtual networks. Accurately predicting individual user influence is vital for understanding social dynamics. Traditional opinion dynamics models, though insightful, require extensive manual calibration and are computationally complex. Recent graph deep learning methods are powerful yet often ignore established social interaction mechanisms. SIGKAN aims to overcome these limitations by introducing a novel approach, combining a sociologically-informed message-passing mechanism based on the BCM and leveraging KANs to update node features. The method also employs an Ordinary Differential Equation (ODE) formulation to constrain the learning process, ensuring adherence to theoretical models. Extensive experiments on Open Academic Graph (OAG), Twitter, Weibo, and Digg datasets demonstrate that SIGKAN outperforms existing baseline models.

## Project Structure
The project consists of the following key components:

- **loader.py**: Contains utilities to load datasets, preprocess data, and prepare input for the model.
- **model.py**: Implements the main SIGKAN model, including SociallogicalInformedConvolution layers and message-passing mechanisms.
- **layers.py**: Defines custom layers, including SociallogicalInformedConvolution and message-passing components.
- **train.py**: Provides training scripts for the model, including dataset handling, loss calculation, and optimization steps.
- **loss.py**: Defines loss functions used during model training.
- **utils.py**: Contains various helper functions and utilities for preprocessing, evaluation, and logging.
- **run.sh**: A script to train the model. This script sets up the environment, executes training, and logs the results.

## Installation
To set up the environment for SIGKAN, follow these steps:

1. **Clone the repository**:
   ```sh
   git clone <repository-url>
   cd SIGKAN
   ```

2. **Create a virtual environment** (recommended):
   ```sh
   python3 -m venv sigkan_env
   source sigkan_env/bin/activate
   ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
To train the SIGKAN model, use the provided `run.sh` script. This script will set up the necessary configuration and execute the training.

```sh
bash run.sh
```

### Parameters
You can customize the training process by modifying the arguments in `run.sh` or directly in `train.py`. Key parameters include:
- **Dataset selection** (e.g., OAG, Twitter, Weibo, Digg)
- **Batch size**, **learning rate**, and other hyperparameters
- **Model architecture** choices such as the use of SIGKAN-Norm or SIGKAN-Att for different similarity calculations

## Model Architecture
SIGKAN introduces several novel components:

- **Sociologically-Informed Message Passing**: A novel mechanism inspired by the Bounded Confidence Model (BCM) to propagate influence across nodes while refining influence weights.
- **Kolmogorov-Arnold Networks (KANs)**: Used to enhance model expressiveness, acting as an MLP for updating node features.
- **ODE Constrained Training**: Integrates theoretical models formulated as ODEs to add constraints during training, ensuring the model's behavior aligns with established social interaction dynamics.
- **SIGKAN-Norm and SIGKAN-Att**: Two model variants using different similarity computation methods, either based on normalized adjacency or self-attention.

## Datasets
- **Open Academic Graph (OAG)**
- **Twitter**
- **Weibo**
- **Digg**

The datasets are preprocessed using `loader.py`, which handles feature extraction, normalization, and graph construction.

## Evaluation
SIGKAN has been evaluated using several key metrics to measure prediction accuracy and generalization. These include:
- **Precision**, **Recall**, **F1-score** for influence prediction.
- **Ablation studies** to demonstrate the contributions of individual components such as BCM and KAN.

## Results
SIGKAN significantly outperforms existing baseline methods on the OAG, Twitter, Weibo, and Digg datasets in predicting social influence. The results show a marked improvement in both the accuracy of individual influence prediction and the model's ability to generalize across different social network datasets.

## Acknowledgments
This work is based on sociological theories of social influence and advanced graph neural network methodologies. We thank the authors and researchers whose works in BCM, KANs, and GNNs have inspired this model.

## License
This project is licensed under the MIT License.

## Contact
For questions, issues, or contributions, please contact [Your Name] at [Your Email].

## Citation
If you use SIGKAN in your research, please cite our paper:

```
@article{sigkan2024,
  title={SIGKAN: Sociologically-Informed Graph Kolmogorov-Arnold Networks for Social Influence Prediction},
  author={Anonymous},
  journal={ACM TheWebConf 2025},
  year={2025}
}
```

