# previsionio-python package examples


- `churn_demo.ipynb` (classification)
- `harddrive_demo.ipynb` (classification, hard drive failure prediction)
- `mercedes_demo.ipynb` (regression, car testing time prediction)

Install required packages (make sure previsionio is already installed)
```
    pip install requirements.txt
```

Start jupyter notebook (make sure jupyter is using the correct environment)
```
    jupyter notebook
```

With anaconda from scratch

```
    conda create --name=prevision-demo python=2.7 --yes
    source activate prevision-demo
    pip install -r requirements .txt
    conda install nb_conda jupyter -n prevision-demo --yes
    cd ..
    python setup.py install
    cd examples
    jupyter notebook
```

