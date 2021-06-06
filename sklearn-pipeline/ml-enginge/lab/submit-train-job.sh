IMAGE_NAME=trainer_image
IMAGE_TAG=latest
IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG


JOB_NAME="JOB_$(date +%Y%m%d)"
JOB_DIR=$ARTIFACT_STORE/jobs/$JOB_NAME
SCALE_TIER=BASIC
REGION=us-central1

DATA_ROOT=$ARTIFACT_STORE/data
TRAINING_FILE_PATH=$DATA_ROOT/training/dataset.csv
VALIDATION_FILE_PATH=$DATA_ROOT/validation/dataset.csv

gcloud ai-platform jobs submit training $JOB_NAME \
--region=$REGION \
--job-dir=$JOB_DIR \
--master-image-uri=$IMAGE_URI \
--scale-tier=$SCALE_TIER \
--config training_app/hptuning_config.yaml \
-- \
--training_dataset_path=$TRAINING_FILE_PATH \
--validation_dataset_path=$VALIDATION_FILE_PATH \
--hptune


#gcloud ai-platform jobs describe $JOB_NAME
#gcloud ai-platform jobs stream-logs $JOB_NAME