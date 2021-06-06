echo DEACTIVATING CONDA ENVIRONMENTS
conda deactivate
conda deactivate
conda deactivate
conda deactivate

echo REMOVING kube ENVIRONMENT
conda remove --name kube --all

echo INSTALLING kube ENVIRONMENT
conda env create -f env.yml