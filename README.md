# quick_draw
### Group
S204618 - Thor Nørgaard Eriksen

S204606 - Liv Hyllinge

S204621 - Kathe H. Schmidt 

S214983 - Søren Mondrup

S214659 - Clara Sofie Christiansen

### Project Overview
This project develops a machine learning model for classifying hand-drawn sketches using Google’s Quick Draw Dataset. The goal is to create an end-to-end MLOps pipeline that effectively processes, classifies and provides predictions for hand-drawn sketches. 

### Framework
The project uses PyTorch Image Models (TIMM) as the primary framework. TIMM provides a comprehensive collection of computer vision models and pre-trained weights. 
The implementation includes:
* Model fine-tuning, for the specific use case. 
* Explore TIMM’s features for improving performance
* Use evaluation tools and metrics, e.g. use Torchmetrics (if compatible with TIMM)

### Data
The Google Quick, Draw! dataset contains millions of sketches across 345 categories, available as either stroke data (json representation of pen strokes, time stamps etc) or rasterized images (28x28 grayscale images). 

Source: Quick, Draw! Dataset

The original approach aims to use the rasterized pixel-data, to focus solely on classifying the drawing. The preprocessing of rasterized data could include normalization of the pixels and apply data augmentation. 
The initial development focuses on 10 categories with approximately 50,000 samples per category, to ensure the model size is manageable. 


### Model:
The model architecture builds on a pre-trained model from the TIMM Framework (maybe ViT). This should ensure a robust performance and acceptable general capabilities for the project.

This project aims to implement computer vision to classify drawings from "Quick, Draw!"


















## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
