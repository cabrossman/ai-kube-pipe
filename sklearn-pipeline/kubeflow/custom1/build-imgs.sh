source ./setup.sh

echo ----------------------
echo BUILDING TRAINER IMAGE
echo ----------------------
gcloud builds submit --timeout 15m --tag $TRAINER_IMG trainer_image

echo ----------------------
echo BUILDING BASE IMAGE
echo ----------------------
gcloud builds submit --timeout 15m --tag $BASE_IMG base_image

