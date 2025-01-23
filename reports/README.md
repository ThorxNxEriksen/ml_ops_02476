# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)      
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)      
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
      One config for Wandb sweep exists.
* [ ] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)


### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] Write API tests for your application and setup continues integration for these (M24)
* [x] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] Create a frontend for your API (M26)

### Week 3

* [x] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [x] Write some documentation for your application (M32)
* [x] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group 49

### Question 2
> **Enter the study number for each member in the group**
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
> Answer:

*s204606, s204618, s204621, s214659, s214983*


### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used the TIMM framework (PyTorch Image Models), a popular library with a wide range of pretrained models for tasks like classification, segmentation, and transfer learning. TIMM is very flexible and gives access to many state-of-the-art models, which made it a great fit for our project.

For our image classification work, we went with the tf_efficientnet_lite0 model. It’s a lightweight version of EfficientNet that’s designed to run efficiently on devices with limited resources, like mobile or edge devices. This model works really well with smaller datasets like ours, the 'Quick, Draw!' dataset, because it balances computational efficiency with strong feature extraction. Since it comes with pretrained weights from ImageNet, we could use transfer learning to reduce training time and still get solid performance with less data. This was especially helpful for the 'Quick, Draw!' dataset, which contains hand-drawn sketches that require the model to pick up on fine details for accurate classification.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We managed our dependencies using a requirements.txt file, which contains a list of all the Python libraries and their versions used in our project. This ensures that every team member works with the same versions of the libraries, avoiding compatibility issues. The dependencies were manually added by team members, so when a group member included a new tool, this was also added to the requirements.txt file. The requirements file includes essential libraries like PyTorch, scikit-learn, and Loguru, along with specific tools like timm for image models. If a new team member were to join, these are the steps to take in order to replicate our setup (in conda):  
1. Install Conda or Miniconda
2. Run: `conda create -n <environment_name> python=X.X` -- creates a new local environment for the project
3. Run: `conda activate <environment_name>` -- activate the environment that was just created
4. Run: `pip install -r requirements.txt` -- installs the requirements to that local environment 

This will ensure that the new member has all the required libraries to be able to continue the development of the project. 
If a new team member were to only develop our classification model, spinning up our dockerfile of the model would replicate the needed environment entirely. 

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We have used the cookiecutter template pretty much as-is. We found it intuitive and fitting for our project, and we filled out the model.py and data.py along with the requirements.txt file first thing. However, we saw fit to split the src folder into two parts, a 'quick_draw' folder for all scripts relating to the data download and preprocessing as well as creating the model, and a 'utils' folder containing the script for logging. In this way we could better separate the different scripts we were developing, and split our scripts into those relevant for the data and model of the project, and helper scripts. Furthermore, we made an 'app' folder for python files and requirements files relating to running the inference model.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

Our approach was to follow the general guidelines of PEP8, and to write comments in the code where necessary. We coded with the goal in mind that the other team members would be able to understand and continue working on our cod. In larger projects, these concepts are greatly important for maintaining consistency and scalability. They help ensure that all developers follow the same coding practices, making the codebase easier to understand and work with. When a new developer joins the project, having clear and consistent practices allows them to quickly learn how to contribute without confusion, regardless of who is guiding their onboarding. Similarly, if a developer leaves the project, these practices make it easier for others to pick up and continue their work. By promoting uniformity and clarity, these concepts reduce misunderstandings and make the project more manageable for everyone involved.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We implemented 3 tests, one test for the model, one for the API and one for the data. The tests are located in the 'test' folder provided by cookiecutter, and are triggered through GitHiub actions, which checks the tests.yml file, which points to the test folder. The `test_data.py` checks if a processed dataset exists, if the load and splitting of the data works, and if the preprocess function works. The `test_model.py` file tests the model using a random input, checking the output shape of the model, verifies whether the classifier layer of the model is trainable, and that the model does not get updated parameters except for in the layer called 'classifier', so that only this layer is trainable. The `test_api.py` tests FastAPI endpoints for root, prediction with/without image, and validates categories list in the backend API.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage of our project is 42%, but even if we had 100% coverage, it wouldn’t guarantee that the code is completely free of errors. Code coverage simply tells us how much of the code was executed during tests — it doesn’t measure the quality or completeness of the tests themselves. For example, tests might not account for edge cases or unusual inputs, and just because a piece of code runs during testing doesn’t mean it’s producing the correct results.

Additionally, there are certain types of issues, like race conditions or hardware-specific bugs, that can’t be caught by typical tests. While high coverage is a great starting point and shows that the code has been exercised thoroughly, it’s not the whole picture. To really trust the code, it’s important to combine high coverage with thoughtful test design, edge case validation, and other testing strategies like integration and stress testing.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

Yes, our whole project was managed through GitHub with branches and pull requests. We tried to follow naming conventions for our branches, starting with the developer's initals and then a short description of what was being done. The initials should also be included in the commit message, as well as an explanation of what was done. After a new branch passed the checks in GitHub (a unit test and a code test) and potential merge conflicts are resolved, the branch is merged with the main branch. After a branch has been merged, we agreed to delete that branch and start on a new one, when developing a new feature, to minimize merge conflicts and ensure no updates to the code are forgotten on a branch that is no longer used. This setup made it easy for more team members to work on the code simultaneously, while also keeping track of new additions, and enabling version control of the code. 

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used DVC to push our data to Google Cloud and for version control. The data can be pulled with the `dvc pull` command. In our project we don't have much use for version control at the moment, as we don't update the dataset. If the project was to be expanded and allow for user-submitted drawings to be added, we could track the development of data and compare the model performance between different datasets. This could potentially help us revert to an earlier dataset, if we got flooded by badly labelled or poorly drawn data.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

Our continuous integration (CI) workflow is designed to ensure code quality and reliability through systematic testing and maintenance. We implemented unit testing via [tests.yaml](https://github.com/ThorxNxEriksen/ml_ops_02476/blob/main/.github/workflows/tests.yaml), which evaluates our data processing, model functionality, and API integration across different environments. Initially, we conducted testing across multiple Python versions (3.11 and 3.12) and operating systems, but this quickly consumed GitHub's computational resources. Therefore, we reduced our testing to Python 3.11 on MacOS, which is a bit more efficient while still performing the needed checks. We intend to revert to the full testing once the project does not have pull requests as frequently as during current development. 
Our linting process uses Ruff through [codecheck.yaml](https://github.com/ThorxNxEriksen/ml_ops_02476/blob/main/.github/workflows/codecheck.yaml) running Ruff to fix simple code formatting errors. Both the unit tests and the linting is set to run on commits to main or when a pull request is created. 

The cookiecutter template also implemented a monthly run of Dependabot which will check if there are updates for the packages in `requirements.txt` and `requirements-dev.txt`. Together, these tests ensure that we have a robust setup while not using too many resources during the development phase. 

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We configured experiments in WandB through a config file called sweep.yml file, which is located in our configs folder and points to our training script train_wandb.py. Experiments are run for the hyperparameters learning rate, epochs and batch size, which the sweep combines to find the minimal validation loss. The sweep can be initalized using the command: `wandb sweep configs/sweep.yaml`, and after the sweep, the results are available in WandB. 


### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

Version control of the config file sweep.yml and train_wandb.py is ensured through GitHub, which tracks any changes to these. WandB tracks each experiment that is run, and since each run has an ID, it can be identified and re-run if needed. The training and validation metrics are also logged for each sweep, so no information is lost while running the experiments. The full model architecture, including weights and biases, is also saved after training, in the 'models' folder as a pth file (a specific file type used for PyTorch for this specific purpose), so that it can be used again. 

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

![image](https://github.com/user-attachments/assets/26773d3b-323f-478d-b985-43d5b5dfaa26)

As seen in the first picture, we have done 4 separate sweeps where we have looked at the loss and accuracy of both the validation and training set. For both of the accuracy graphs, the accuracy steadily increases, while the loss decreases for both the validation and training set, indicating that new and better combinations of the hyperparameters are found at each step.

![image](https://github.com/user-attachments/assets/caa3ef6a-c6f6-44e7-87fa-a4b1b634fc79)

The second image shows how the values of the hyperparameters change for each experiment. The 4 experiments are visualized in the second image, which shows what different combinations of these hyperparameters are being used for a given experiment. These hyperparameters are greatly important for model performance. The learning rate controls the size of updates to the model's weights, so a large learning rate might take steps that are too large and miss good values in between, while a smaller learning rate is much slower. The batch size determines the number of training samples each training iteration to compute the gradient and update the weights, so small batch sizes can introduce more noise, and while larger batches lead to more stable updates, they also require much more memory. Finally, number of epochs is the times that the entire dataset is put through the model. Too few might lead to overfitting, while too many can lead to overfitting. Getting these hyperparameters right is crucial for having a robust and well-performing model. On the other hand we have not sacrificed much time for this step as we just wanted to prove that it works and focus on new, exiting functionalities more focused on the Operations part. 


### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

In our project, we developed two essential Docker images to containerize our Quick Draw classification model. The first image handles model training using our EfficientNet architecture, while the second serves our FastAPI inference endpoint. The training container can be executed (local) with customizable hyperparameters using: `docker run train:latest --lr 0.001 --batch_size 32 --epochs 1`. For deployment (local), we run our API container using: `docker run --rm -p 8000:8000 api_image:latest`, which exposes the FastAPI application for sketch classification predictions. Link to docker file: [train.dockerfile](https://github.com/ThorxNxEriksen/ml_ops_02476/blob/main/dockerfiles/train.dockerfile)

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

The preferred debugging of all group members were to use the built-in debugger in Visual Studio code, as this is intuitive and easy to use. It worked well, and some team members also used GitHub co-pilot in addition to the built in debugger, to help resolve potential issues with the code. We were also good at helping each other in the group, if neither the debugger nor co-pilot could help solve the issue. We eventually did a primitive profiling run, which focused on the top 20 functions that took the most time. But since our model and setup is relatively simple, we did not have much time to gain by trying to improve these models, so we did not go further with this. 


## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

Cloud build - Builds the docker images on any commit to the `main` in the Github repo and pushes them to Artifact registry.

Artifact registry - Stores the images created by Cloud build

Engine - Runs a VM which can then run the containers based on the image retrieved from Artifact registry.

Bucket - Stores data sets and (ideally) the latest model file generated from training in the Engine.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

The compute engine is the most important part of the Google Cloud Platform. We have used the Compute engine to create all the instances of our virtual machines (and let me tell you there have been a few versions...) from our docker images in the Google Cloud Platform Artifacts. We have used europe-west1 as the zone for hosting. Most virtual machine instances has been n1-standard-4 with increased storage. This choice was based on two factors:
1. The n1 is one of the cheapest options
2. The choice of saving the data in the docker container and the mere size of the requirements file resulted in memory errors for a standard n1.

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![image](figures/artifacts.png)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![image](figures/cloud_buckets.png)

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We managed to train the model on Engine, by building the `train_gcloud.dockerfile` in Cloud Build using a trigger and pushing it to Artifact Registry. Running the container in the VM, we had the model train on the data and connect to WandB. However, we had trouble connecting to our data bucket on mount, and therefore all the data is downloaded from an external server using the `quickdraw`-package. The model should ideally be saved in the data bucket, but is ATM only available via WandB.
Since the current scope of the project isn't to add data progressively, we decided to move on. 

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We successfully implemented a backend API for our QuickDraw classification model using FastAPI. We created a POST endpoint `/predict` that accepts image uploads in any common format (PNG, JPEG), converts them to grayscale, and applies the same preprocessing pipeline as our training data - resizing to 224x224 pixels using torchvision transforms. The API returns a JSON response containing both the predicted drawing category and a confidence score between 0 and 1.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 24 fill here ---

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

For unit testing, we used pytest with FastAPI's TestClient to test API endpoints. We created mock images using PIL for testing the prediction endpoint, ensuring consistent test conditions. The tests covered basic functionality, error handling, and input validation.

For load testing, we implemented Locust with a script simulating users making requests to both root and prediction endpoints. The script generated a 224x224 grayscale dummy image per user, reusing it across requests with 1-3 second intervals between actions. Our load testing revealed scalability: with 500 concurrent users and 50 users/second spawn rate, the API maintained 100% success rate. However, we identified two breaking points: first, when ramping up to 1000 users at 50 users/second, we observed 5% failure rate around 700 concurrent users. Second, with 500 users but faster spawn rate (100 users/second), failures began appearing at 300-400 concurrent users


### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We did not implement monitoring of our deployed model. We would have liked to, as we could then observe our model by tracking performance metrics like accuracy and per-class precision, detecting data distribution shifts, observing model behaviour such as misclassification patterns, and checking system health indicators like resource utilization and prediction error rates. These monitoring approaches would help us understand model performance, identify potential degradation, and inform timely retraining or fine-tuning decisions. By implementing continuous monitoring, we can proactively maintain the model's reliability and ensure it remains effective as data and classification requirements evolve over time.


## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

In total, we spent $11.3 spread across various Google Cloud services:
	Cloud Run	$0.00
	Compute Engine	$10.37
	Cloud Storage	$0.16
	Networking	$0.28
	Artifact Registry	$0.46
Spending the majority on compute makes sense. Even if our model is very quick to train we have spun up a bunch of different images while getting them to work. If we wanted to permanently run the API we would expect this to increase
Since our dataset is 2GB and we only used one version, the load on the storage is low. This would also increase once we start updating our dataset and keeping multiple versions.
Working in the cloud was a little frustrating, because very small changes in the local setups can take up to 10 minutes to show in the cloud build. Having said that, we also had bigger celebrations when the cloud ran because it felt a bit more real!

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

We implemented a front-end for our API using Streamlit. This worked by making a docker image containing all the files and code used for the inference API and the Streamlit page, which was then deployed. We wanted a nicer interface for the API, instead of using the interface provided by FastAPI, and so we decided to make this in Streamlit. In the Streamlit appliciation, a user can upload an image of any format, and the API will use our model to classify the image, return a classification and a confidence score. The Streamlit page can be seen in the image below: 
![image](https://github.com/user-attachments/assets/eac61a5a-e99f-495d-a92a-64372db4b057)


We also implemented drift detection using Evidently, where we compared our training set to out test set, to check if the test set is a good representation of our training set (and because we did not have any obvious way to split out data in current and historical). The metrics used for drift detection are number of pixels, mean intensity and labels. We discovered no data drift, which can be seen below: 
![image](https://github.com/user-attachments/assets/fa2010f5-b7ae-47f8-a625-d310437e71d6)




### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:
![image](https://github.com/user-attachments/assets/45b321a2-3373-4827-82da-955e599f411c)

Developers can code on their own laptops and commit and push changes to the GitHub repository for version control. GitHub Actions include unit tests, Ruff for linting, and Dependabot for updating dependency versions in requirements.txt which automatically activate at each push/PR to main. Pushing to any branch triggers a cloud build that creates a Docker image, which is stored in Google Cloud Platform (GCP) Artifacts. A virtual machine can be initialized from the Docker image to train the model in the cloud. Experiment logs are saved via Weights & Biases (wandb), both those run locally and in the cloud. Wandb both logs statistics such as accuracy and loss but also saves the model in its Artifacts folder (with version control). Additionally, another Docker image can be initialized for the FastAPI, which uses Streamlit as the frontend.

The data is version controlled using DVC and is connected to the storage in the Google GCP data bucket.

Users have three options for interacting with our program:
- Pull the latest image for training or API.
- Clone the code or read the documentation for classes and functions.
- Query the server with an image for classification. The user will receive the most probable class of that image. 

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

The biggest challenge we faced during the project was working with Google Cloud Platform (GCP). Setting up and integrating the modules proved to be very difficult. Each small change required rebuilding the Docker image, which resulted in significant waiting times. We encountered numerous errors along the way, but we managed to resolve most of them through troubleshooting with Google and ChatGPT.

However, the biggest challenge was mounting the data. According to Nicki's answer on Slack, this process was supposed to be automatic, which added to our confusion when it didn't work as expected. After spending two days trying to resolve this issue, we ultimately decided that it wasn't worth further delays. Instead, we opted to download and process the data within the container on the cloud.

Running the API and frontend on the cloud presented another set of challenges. We encountered several issues related to configuration and deployment, which required extensive debugging and adjustments. Despite these hurdles, we managed to get the system up and running, but it took a significant amount of time and effort.

As mentioned in other questions: Working in the cloud was frustrating but we also had bigger celebrations when the cloud ran because it felt a bit more real!




### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

There was an emphasis from the group that everyone should be involved in the code, especially in understanding the general overview and elements of the code. Generally, the tasks that were ongoing included Docker, code typing, and writing the report which everyone contributed to.


| Student nr. | Contribution |
|----------|----------|
| s204606    | ML model, Documentation |
| s204618    | Fronted API, Streamlit app, API integration into cloud   |
| s204621    | Backend API, Load test, Test of API   |
| s214659    | Cloud, WandB, ML model  |
| s214983    | Cloud, Github workflow, unit test  |

