
./setup.sh

echo BUILDING TRAINER IMAGE
gcloud builds submit --timeout 15m --tag $TRAINER_IMG trainer_image


echo BUILDING BASE IMAGE
gcloud builds submit --timeout 15m --tag $BASE_IMG base_image

