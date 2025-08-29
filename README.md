# Titanic Kaggle Competition

This project is my work participating in the Kaggle competition "[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)". This project contains the Jupyter notebook stripped of its outputs and a corresponding Python script.

## Requirements

The full list of Python packages I used in my Conda environment can be found in [environment.yml](./environment.yml). Not all of them are used in this project, this is just the collection of all the machine learning and data science packages I find useful.

To create an environment from the environment.yml file, follow the instructions in the [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file). In short, the command is:

```bash
conda env create -f environment.yml
```

## Converting Between Notebook and Python File

To convert between Jupyter notebook and Python script, I use the [Jupytext](https://jupytext.readthedocs.io/en/latest/) package. It can be install via Pip:

```bash
pip install jupytext
```

Conversion from notebok to Python script:

```bash
jupytext --to py:percent titanic.ipynb -o titanic.py
```

Conversion from Python script to notebook:

```bash
jupytext --to ipynb titanic.py -o titanic.ipynb
```

Also, for clearing notebook outputs, I use the [nbstripout](https://github.com/kynan/nbstripout) command:

```bash
nbstripout titanic.ipynb
```

## Contact

If you have any questions or feedback, feel free to connect with me on LinkedIn at [linkedin.com/in/daniel-di-giovanni/](https://www.linkedin.com/in/daniel-di-giovanni/) or send me an email at [dannyjdigio@gmail.com](mailto:dannyjdigio@gmail.com).
