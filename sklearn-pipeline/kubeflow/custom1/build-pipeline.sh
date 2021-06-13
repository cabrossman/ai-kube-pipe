source ./setup.sh

PIPELINE_NAME=pricing_pipeline

### OTHERS
EXPERIMENT_NAME=Pricing_Classifier_Training
RUN_ID="RUN_$(date +%Y%m%d_%H%M%S)"
SOURCE_TABLE=aitest.pricing
DATASET_ID=splits
EVALUATION_METRIC=accuracy
EVALUATION_METRIC_THRESHOLD=0.69
MODEL_ID=pricing_classifier
VERSION_ID=v01
GCS_STAGING_PATH=$ARTIFACT_STORE/staging



### Convert pipeline to YAML
dsl-compile --py pipeline/$PIPELINE_NAME.py --output $PIPELINE_NAME.yaml

### Check if PIPELINE already has value - you cant upload a pipeline with same name
PIPELINE_ID=$(kfp --endpoint $ENDPOINT pipeline list | grep -w $PIPELINE_NAME | tr -s ' ' | cut -d ' ' -f 2 | xargs)
if [ -z "$PIPELINE_ID" ]
then
    echo -------------------
    echo UPLOADING PIPELINE
    echo -------------------
    kfp --endpoint $ENDPOINT pipeline upload -p $PIPELINE_NAME $PIPELINE_NAME.yaml
    PIPELINE_ID=$(kfp --endpoint $ENDPOINT pipeline list | grep -w $PIPELINE_NAME | tr -s ' ' | cut -d ' ' -f 2 | xargs)
else 
    echo -------------------
    echo PIPELINE ALREADY EXISTS
    echo -------------------
fi

echo -------------------
echo UPLOADING A RUN
echo -------------------
kfp --endpoint $ENDPOINT run submit \
    -e $EXPERIMENT_NAME \
    -r $RUN_ID \
    -p $PIPELINE_ID \
    project_id=$PROJECT_ID \
    gcs_root=$GCS_STAGING_PATH \
    region=$REGION \
    source_table_name=$SOURCE_TABLE \
    dataset_id=$DATASET_ID \
    evaluation_metric_name=$EVALUATION_METRIC \
    evaluation_metric_threshold=$EVALUATION_METRIC_THRESHOLD \
    model_id=$MODEL_ID \
    version_id=$VERSION_ID