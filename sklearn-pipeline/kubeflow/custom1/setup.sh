
TRAINER_IMG=gcr.io/$PROJECT_ID/pricing:latest
BASE_IMG=gcr.io/$PROJECT_ID/base_image:latest

export USE_KFP_SA=False
export BASE_IMAGE=$BASE_IMG
export TRAINER_IMAGE=$TRAINER_IMG
export COMPONENT_URL_SEARCH_PREFIX=https://raw.githubusercontent.com/kubeflow/pipelines/0.2.5/components/gcp/
export RUNTIME_VERSION=1.15
export PYTHON_VERSION=3.7


