source ./setup.sh


echo ----------------------
echo BUILDING TRAINER IMAGE
echo ----------------------
gcloud builds submit --timeout 15m --tag $TRAINER_IMAGE trainer_image

echo ----------------------
echo BUILDING BASE IMAGE
echo ----------------------
gcloud builds submit --timeout 15m --tag $BASE_IMAGE base_image

