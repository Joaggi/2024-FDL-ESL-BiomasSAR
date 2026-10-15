# FDL-BiomassSar

<p>
  1.<a href="https://arxiv.org/abs/2411.12897"><img src="http://img.shields.io/badge/Paper-PDF-brightgreen.svg"></a>
  2.<a href="https://arxiv.org/abs/2412.00903"><img src="http://img.shields.io/badge/Paper-PDF-brightgreen.svg"></a>
  2.<a href="https://arxiv.org/abs/2409.05636"><img src="http://img.shields.io/badge/Paper-PDF-brightgreen.svg"></a>
  <a href="https://github.com/Joaggi/Robust-kernels-for-robust-location-estimation/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  </a>
</p>

Tree Species Classification using Machine Learning and 3D Tomographic SAR -- a case study in Northern Europe

Tree species classification plays an important role in nature conservation, forest inventories, forest management, and the protection of endangered species. Over the past four decades, remote sensing technologies have been extensively utilized for tree species classification, with Synthetic Aperture Radar (SAR) emerging as a key technique. In this study, we employed TomoSense, a 3D tomographic dataset, which utilizes a stack of single-look complex (SLC) images, a byproduct of SAR, captured at different incidence angles to generate a three-dimensional representation of the terrain. Our research focuses on evaluating multiple tabular machine-learning models using the height information derived from the tomographic image intensities to classify eight distinct tree species. The SLC data and tomographic imagery were analyzed across different polarimetric configurations and geosplit configurations. We investigated the impact of these variations on classification accuracy, comparing the performance of various tabular machine-learning models and optimizing them using Bayesian optimization. Additionally, we incorporated a proxy for actual tree height using point cloud data from Light Detection and Ranging (LiDAR) to provide height statistics associated with the model's predictions. This comparison offers insights into the reliability of tomographic data in predicting tree species classification based on height. 


## Installation

Install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html) and then run the following commands to create the fdl-sar-env environment:

```bash
conda env create -f environment.yml

conda activate fdl-biomass-env
```

Next, install the package:

```bash
pip install -e .
```

or if you want development dependencies as well:

```bash
pip install -e .[dev]
```

### Optional, but highly recommended

Install [pre-commit](https://pre-commit.com/) by running the following command to automatically run code formatting and linting before each commit:

```bash
pre-commit install
```

If using pre-commit, each time you commit, your code will be formatted, linted, checked for imports, merge conflicts, and more. If any of these checks fail, the commit will be aborted.

## Adding a new package

To add a new package to the environment, open pyproject.toml file and add the package name to "dependencies" list. Then, run the following command to install the new package:

```bash
pip install -e . # or .[dev]
```

## Cache data
Data will be stored in `.cache` inside the folder from where you run the script. If you want to change the cache_dir location, you can set the environment variable `CACHE_DIR` to the desired location. To do so, create a `.env` file and add inside it the following line:

```bash
CACHE_DIR=/path/to/cache/dir
```


## 📖 Cite as 


```
@article{grace2024tree,
  title={Tree Species Classification using Machine Learning and 3D Tomographic SAR--a case study in Northern Europe},
  author={Grace, Colverd and Laura, Schade and Jumpei, Takami and Karol, Bot and Joseph, Gallego},
  journal={arXiv preprint arXiv:2411.12897},
  year={2024}
}

@article{colverd20243d,
  title={3D-SAR tomography and machine learning for high-resolution tree height estimation},
  author={Colverd, Grace and Takami, Jumpei and Schade, Laura and Bot, Karol and Gallego-Mejia, Joseph A},
  journal={arXiv preprint arXiv:2409.05636},
  year={2024}
}

@article{colverd2024tomographic,
  title={Tomographic SAR Reconstruction for Forest Height Estimation},
  author={Colverd, Grace and Takami, Jumpei and Schade, Laura and Bot, Karol and Gallego-Mejia, Joseph A},
  journal={arXiv preprint arXiv:2412.00903},
  year={2024}
}
```  
