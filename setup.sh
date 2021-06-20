echo DEACTIVATING CONDA ENVIRONMENTS
conda deactivate
conda deactivate
conda deactivate
conda deactivate

echo REMOVING kube ENVIRONMENT
conda remove --name kube --all

echo INSTALLING kube ENVIRONMENT
conda env create -f env.yml
conda create -n kube python=3.7.10
#conda activate kube
#pip install -r requirements-tfx.txt