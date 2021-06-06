
DATASET_ID=covertype_dataset
DATASET_LOCATION=US
TABLE_ID=covertype
DATA_SOURCE=gs://workshop-datasets/covertype/small/dataset.csv
SCHEMA=Elevation:INTEGER,Aspect:INTEGER,Slope:INTEGER,Horizontal_Distance_To_Hydrology:INTEGER,Vertical_Distance_To_Hydrology:INTEGER,Horizontal_Distance_To_Roadways:INTEGER,Hillshade_9am:INTEGER,Hillshade_Noon:INTEGER,Hillshade_3pm:INTEGER,Horizontal_Distance_To_Fire_Points:INTEGER,Wilderness_Area:STRING,Soil_Type:STRING,Cover_Type:INTEGER
REGION=us-central1

echo CREATE DATASET
bq --location=$DATASET_LOCATION --project_id=$PROJECT_ID mk --dataset $DATASET_ID

echo LOAD DATA INTO DATASET
!bq --project_id=$PROJECT_ID --dataset_id=$DATASET_ID load \
--source_format=CSV \
--skip_leading_rows=1 \
--replace \
$TABLE_ID \
$DATA_SOURCE \
$SCHEMA

echo CREATE training dataset table
bq query \
-n 0 \
--destination_table covertype_dataset.training \
--replace \
--use_legacy_sql=false \
'SELECT * \
FROM `covertype_dataset.covertype` AS cover \
WHERE \
MOD(ABS(FARM_FINGERPRINT(TO_JSON_STRING(cover))), 10) IN (1, 2, 3, 4)' 

echo CREATE validation dataset table
bq query \
-n 0 \
--destination_table covertype_dataset.validation \
--replace \
--use_legacy_sql=false \
'SELECT * \
FROM `covertype_dataset.covertype` AS cover \
WHERE \
MOD(ABS(FARM_FINGERPRINT(TO_JSON_STRING(cover))), 10) IN (8)' 


echo EXTRACT to GCS
!bq extract \
--destination_format CSV \
covertype_dataset.training \
$TRAINING_FILE_PATH

!bq extract \
--destination_format CSV \
covertype_dataset.validation \
$VALIDATION_FILE_PATH