# Quick Draw Sketch Classification: An MLOps Implementation
### Group
s204618 - Thor Nørgaard Eriksen

s204606 - Liv Hyllinge

s204621 - Kathe H. Schmidt 

s214983 - Søren Mondrup

s214659 - Clara Sofie Christiansen

## Project Description
This project aims to develop a comprehensive machine learning pipeline for the classification of hand-drawn sketches utilizing Google's "Quick, Draw!" Dataset. The primary objective is to construct an end-to-end MLOps infrastructure that encompasses data processing, model training, deployment, and monitoring while adhering to modern MLOps practices and principles.

### Overall Project Goal
The fundamental goal is to create a robust and scalable system capable of accurately classifying hand-drawn sketches across multiple categories. Beyond mere classification, this project seeks to implement a complete MLOps pipeline that demonstrates industry-standard practices in machine learning deployment. Key objectives include:

* Development of an efficient data pipeline for processing and managing sketch data
* Implementation of a model training and evaluation framework
* Creation of a scalable deployment infrastructure
* Integration of monitoring and maintenance systems
* Establishment of continuous integration and deployment practices


### Framework
The project will primarily utilize PyTorch Image Models (TIMM) as its core framework, leveraging its extensive collection of state-of-the-art computer vision models and pre-trained weights. TIMM's integration will span several crucial aspects:

* Model Architecture Selection: Utilizing TIMM's diverse model zoo to identify and implement appropriate architectures for sketch classification
* Transfer Learning: Leveraging pre-trained weights while adapting the models for sketch-specific features
* Training Optimization: Implementing TIMM's advanced training techniques and optimization strategies
* Performance Metrics: Integrating TIMM's evaluation tools with additional metrics from Torchmetrics for comprehensive model assessment

### Data
The project will utilize the Google "Quick, Draw!" Dataset, a comprehensive collection of hand-drawn sketches comprising millions of samples across 345 categories. Our initial approach focuses on the rasterized version of the dataset, which provides 28x28 grayscale images. This decision allows us to concentrate on pure visual classification aspects while maintaining computational efficiency.
Data processing will involve:

* Initial focus on 10 selected categories, each containing approximately 50,000 samples
* Implementation of preprocessing pipelines, including normalization and augmentation
* Development of efficient data loading and batching mechanisms


### Model
The model architecture will be based on pre-trained models available through the TIMM framework, with particular consideration given to Vision Transformers (ViT) due to their strong performance in image classification tasks. The selection criteria will include:

* Model complexity and computational requirements
* Adaptation capability for sketch-specific features
* Scalability considerations for future expansion
* Balance between accuracy and inference speed

The initial implementation will focus on fine-tuning these pre-trained models for sketch classification, with potential exploration of architectural modifications to better suit the specific characteristics of hand-drawn sketches.


















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
