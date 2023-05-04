# OFA³: Automatic Selection of the Best Non-dominated Sub-networks for Ensembles

# The framework

**TL;DR**: The OFA³ is a method to automatically select a subset of architectures among a population of neural networks in a multi-objective perspective, considering the conflicting objective functions (accuracy and latency/FLOPs, for example) to form efficient ensembles.

The steps to reproduce the results of this work are described as follows (all notebooks needed are provided [here](https://github.com/ito-rafael/once-for-all-3/tree/main/jupyter-notebooks)).

1. Perform the [OFA²](https://arxiv.org/abs/2303.13683) Multi-Objective Optimization NAS (Neural Architectural Search) search method on the [OFA](https://arxiv.org/abs/1908.09791) search space. The output of this stage should be 100 efficient sub-networks of the OFA supernetwork, each with a different trade-off among the objective functions (accuracy and latency).
2. For each of these 100 neural networks, calculate the top-5 class predictions and the respective top-5 probabilities by appending a Softmax layer on the structure of these architectures.
3. Run the OFA³ algorithm for selecting a subset of neural networks among the population of 100 (obtained with the OFA² search) to form efficient ensembles.

# Related works

# OFA (Once-for-All)
  - Description: Train One Network and Specialize it for Efficient Deployment
  - Paper: [arXiv](https://arxiv.org/abs/1908.09791)
  - Git Repository: https://github.com/mit-han-lab/once-for-all

# OFA²
  - Description: Train one network, Search once, Deploy in many scenarios
  - Paper: [arXiv](https://arxiv.org/abs/2303.13683)
  - Git Repository: https://github.com/ito-rafael/once-for-all-2

# Dataset

The dataset used for this project is the same used in the OFA and OFA² NAS algorithms, that is, the ILSVRC (ImageNet-1k). This dataset contains images of 1,000 categories and is divided as follows:

| Set description | Number of samples | Contain labels? |
|:---------------:|:-----------------:|:---------------:|
| Training set    | 1,281,167         | yes             |
| Validation set  | 50,000            | yes             |
| Test set        | 100,000           | no              |

Additionally, we generated a subset of the training set containing 50,000 images (50 of each class) in order to perform the OFA³ selection algorithm, while keeping the validation set untouched for fair performance comparison with other methods. To generate this subset of 50k images, one can first run the [<ins>dataset/create_dirs.sh</ins>](https://github.com/ito-rafael/once-for-all-3/blob/main/dataset/create_dirs.sh) script to generate the 1,000 directories corresponding to each image category, and then run the [<ins>dataset/create_train_subset_50k.sh</ins>](https://github.com/ito-rafael/once-for-all-3/blob/main/dataset/create_train_subset_50k.sh) script, to copy the first 50 images of each category in the respective directory.

For more information related to the dataset, please check the original [ImageNet website](https://www.image-net.org/update-mar-11-2021.php).

## Additional information

- The [OFA](https://arxiv.org/abs/1908.09791) supernetwork was trained with the training set of the ILSVRC (1,281,167 images).
- The [OFA²](https://arxiv.org/abs/2303.13683) search uses the efficiency predictors (latency or FLOPS) and the accuracy predictor of the OFA framework. Since the accuracy predictor was trained with a subset of the training set, the OFA² is inherently also tangled to the training set of ILSVRC.
- The OFA³ (ours) method uses a subset of the training set with 50k images to guide our selection algorithm. 
- This way, we can leave the validation set untouched for all three approaches (OFA/OFA²/OFA³), leaving a fair way of performance comparison between these methods.

# Jupyter notebooks
There are several Jupyter notebooks, each responsible for a specific task and full of details about the implementation. They are divided into three main groups: the notebook related to the OFA² search, the notebooks that use the architectures found by the OFA² search and are used as a preparation for the OFA³ search, and finally, the notebooks properly related to the OFA³ search.

## OFA²
- [<ins>ofa2.ipynb</ins>](https://github.com/ito-rafael/once-for-all-3/blob/main/jupyter-notebooks/ofa2.ipynb): This notebook runs the OFA² search under the search space of the OFA supernetwork. Three EMOA (Evolutionary Multi-Objective Optimization Algorithm) are compared: NSGA-II, SMS-EMOA and SPEA. We perform three searches with three different RNG seeds for each algorithm. The output of this notebook is 100 efficient sub-networks of the OFA supernetwork, each with a different trade-off among the objective functions (accuracy and latency).

## OFA²·⁵
This category of notebooks is used as a preparation for the OFA³ selection algorithm. Each of these notebooks takes the 100 architectures obtained as the output of the OFA² search method and evaluates them on a set of images from ILSVRC (ImageNet-1k).
<br>
The outputs of each notebook are 2 CSV tables for each architecture (thus 200 CSV files in total), with the following naming structure:
  1. **model_[0-9][0-9]_class.csv**: This table contains the top-5 predicted output class for the images being evaluated.
  2. **model_[0-9][0-9]_prob.csv**: This table contains the respective top-5 predicted probabilities (softmax output) for the images being evaluated.
<br><br>
Next, there is the list of notebooks followed by the set of images used to generate the aforementioned tables.
- [<ins>ofa2_5-generate-prob-table-train-1M.ipynb</ins>](https://github.com/ito-rafael/once-for-all-3/blob/main/jupyter-notebooks/ofa2_5-generate-prob-table-train-1M.ipynb): This notebook evaluates the 100 architectures obtained from the OFA² search (EMOA: NSGA-II) on the full training set of the ILSVRC (ImageNet-1k), totalizing 1,281,167 images.
- [<ins>ofa2_5-generate-prob-table-train-subset-50k.ipynb</ins>](https://github.com/ito-rafael/once-for-all-3/blob/main/jupyter-notebooks/ofa2_5-generate-prob-table-train-subset-50k.ipynb): This notebook evaluates the 100 architectures obtained from the OFA² search (EMOA: NSGA-II) on a subset of the training set of the ILSVRC (ImageNet-1k), totalizing 50,000 images.
- [<ins>ofa2_5-generate-prob-table-val-50k.ipynb</ins>](https://github.com/ito-rafael/once-for-all-3/blob/main/jupyter-notebooks/ofa2_5-generate-prob-table-val-50k.ipynb): This notebook evaluates the 100 architectures obtained from the OFA² search (EMOA: NSGA-II) on the full validation set of the ILSVRC (ImageNet-1k), totalizing 50,000 images.

## OFA³
- [<ins>ofa3-max-latency.ipynb</ins>](https://github.com/ito-rafael/once-for-all-3/blob/main/jupyter-notebooks/ofa3-max-latency.ipynb): This notebook runs the OFA³ selection method over the 100 architectures obtained from OFA² search using the maximum latency criteria. The output of this notebook is 100 efficient ensembles.
- [<ins>ofa3-summed-latency.ipynb</ins>](https://github.com/ito-rafael/once-for-all-3/blob/main/jupyter-notebooks/ofa3-summed-latency.ipynb): This notebook runs the OFA³ selection method over the 100 architectures obtained from OFA² search using the summed latency criteria. The output of this notebook is 100 efficient ensembles.
- [<ins>ofa3-plot.ipynb</ins>](https://github.com/ito-rafael/once-for-all-3/blob/main/jupyter-notebooks/ofa3.ipynb): This notebook plot the results from the OFA³ selection method for both the summed and maximum latency scenarios. There are also several comparisons with the OFA and OFA² architectures.
