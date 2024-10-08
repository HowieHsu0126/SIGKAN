# SIGKAN: Sociologically-Informed Graph Kolmogorov-Arnold Networks for Social Influence Prediction

> **TL;DR:** This paper introduces SIGKAN, a novel framework that integrates sociological principles into graph-based deep learning models for predicting social influence.

## Abstract
Social influence plays a crucial role in both offline interactions and virtual networks. Accurately predicting individual user influence is vital for understanding social dynamics. Traditional opinion dynamics models, while insightful, require extensive manual calibration and are computationally complex. Recent graph deep learning methods, although powerful, often ignore established social interaction mechanisms. SIGKAN addresses these limitations by introducing a sociologically-informed message-passing mechanism based on the Bounded Confidence Model (BCM) and leveraging Kolmogorov-Arnold Networks (KANs) for feature updates. The method also employs an Ordinary Differential Equation (ODE) formulation to constrain the learning process, ensuring adherence to theoretical models. Extensive experiments on Open Academic Graph (OAG), Twitter, Weibo, and Digg datasets demonstrate that SIGKAN significantly outperforms existing baselines.

## Project Structure
The project comprises the following key components:

- **loader.py**: Utilities for loading datasets, preprocessing data, and preparing input for the model.
- **model.py**: Implementation of the main SIGKAN model, including SociallogicalInformedConvolution layers and message-passing mechanisms.
- **layers.py**: Custom layers such as SociallogicalInformedConvolution and message-passing components.
- **train.py**: Training scripts, including dataset handling, loss calculation, and optimization steps.
- **loss.py**: Loss functions used during model training.
- **utils.py**: Helper functions for preprocessing, evaluation, and logging.
- **run.sh**: Script to set up the environment, execute training, and log results.

## Installation
To set up the environment for SIGKAN:

1. **Clone the repository**:
   ```sh
   git clone https://anonymous.4open.science/anonymize/SIGKAN-3651
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
To train the SIGKAN model, use the provided `run.sh` script:

```sh
bash run.sh
```

### Parameters
Customize the training process by modifying arguments in `run.sh` or directly in `train.py`. Key parameters include:
- **Dataset selection**: OAG, Twitter, Weibo, Digg
- **Batch size**, **learning rate**, and other hyperparameters
- **Model architecture**: SIGKAN-Norm or SIGKAN-Att for different similarity calculations

## Model Architecture
SIGKAN introduces several novel components:

- **Sociologically-Informed Message Passing**: Inspired by the Bounded Confidence Model (BCM) to propagate influence across nodes while refining influence weights.
- **Kolmogorov-Arnold Networks (KANs)**: Enhance model expressiveness, acting as a Multi-Layer Perceptron (MLP) for updating node features.
- **ODE Constrained Training**: Integrates theoretical models formulated as ODEs to add constraints during training, ensuring the model's behavior aligns with established social interaction dynamics.
- **SIGKAN-Norm and SIGKAN-Att**: Two model variants using different similarity computation methods, either based on normalized adjacency or self-attention.

## Datasets
- **Open Academic Graph (OAG)**
- **Twitter**
- **Weibo**
- **Digg**

The datasets are preprocessed using `loader.py`, which handles feature extraction, normalization, and graph construction.

### Download Links
Download the preprocessed datasets from [OneDrive](https://1drv.ms/f/s!An4lcD8a80_7gzdLaanNUThTWwmy), [Dropbox](https://www.dropbox.com/s/y1iokawi33mn87y/DeepInf.tar.gz?dl=0), [Google Drive](https://drive.google.com/open?id=1qBIVdwkKcnOGZnXHcIizzW4_bUekRgC6), or [Baidu Pan](https://pan.baidu.com/s/1YX3cHYaK_7UuX4qEnqgo9w) (password: 242g).

For un-preprocessed data, download them from the following links:

- [Digg dataset](https://www.isi.edu/~lerman/downloads/digg2009.html)
- [Twitter dataset](https://snap.stanford.edu/data/higgs-twitter.html)
- [OAG dataset](https://www.openacademic.ai/oag/)
- [Weibo dataset](https://www.aminer.cn/influencelocality)

## Evaluation
SIGKAN has been evaluated using several key metrics to measure prediction accuracy and generalization:
- **Precision**, **Recall**, **F1-score** for influence prediction.
- **Ablation studies** to demonstrate the contributions of individual components such as BCM and KAN.

## Results
SIGKAN significantly outperforms existing baseline methods on the OAG, Twitter, Weibo, and Digg datasets in predicting social influence. The results demonstrate a substantial improvement in both accuracy and generalization capabilities across different social network datasets.

## Acknowledgments
This work is based on sociological theories of social influence and advanced graph neural network methodologies. We acknowledge the works that inspired this model, including **DeepInf** ([GitHub Repository](https://github.com/xptree/DeepInf)) and **Efficient KAN** ([GitHub Repository](https://github.com/Blealtan/efficient-kan)).

## License
This project is licensed under the MIT License.

## Citation
If you use SIGKAN in your research, please cite our paper:

```
@article{sigkan2025,
  title={SIGKAN: Sociologically-Informed Graph Kolmogorov-Arnold Networks for Social Influence Prediction},
  author={Anonymous},
  journal={TheWebConf},
  year={2025}
}
```