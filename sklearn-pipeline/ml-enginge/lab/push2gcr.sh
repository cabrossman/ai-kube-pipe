IMAGE_NAME=trainer_image
IMAGE_TAG=latest
IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG

gcloud builds submit --tag $IMAGE_URI training_app