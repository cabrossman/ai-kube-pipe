
DATASET_ID=aitest
DATASET_LOCATION=US
JOB_DIR_ROOT=$ARTIFACT_STORE/jobs
TRAINING_FILE_PATH=$JOB_DIR_ROOT/training/pricing.csv
VALIDATION_FILE_PATH=$JOB_DIR_ROOT/validation/pricing.csv

echo CREATE DATASET
bq --location=$DATASET_LOCATION --project_id=$PROJECT_ID mk --dataset $DATASET_ID

echo LOAD DATA INTO DATASET
bq query \
-n 0 \
--destination_table $DATASET_ID.pricing \
--replace \
--use_legacy_sql=false \
'select * from `therealreal.com:api-project-837871631769.tryolabs.item_history_fixed` where available_at >= TIMESTAMP("2020-01-01")'

echo CREATE TRAINING dataset table
bq query \
-n 0 \
--destination_table $DATASET_ID.training \
--replace \
--use_legacy_sql=false \
"select \
     latest_customer_price, \
     latest_promotion_rate, \
     percent_available, \
     latest_base_price, \
     original_price, \
     total, \
     item_views, \
     items_sold, \
     average_retail, \
     designer_id, \
     taxon_permalink, \
     condition, \
     merchandising_category, \
     case when change_type = 'sale' then 1 else 0 end as sold \
from aitest.pricing \
where MOD(ABS(FARM_FINGERPRINT(cast(item_id as STRING))), 100) in (1,2,3) \
and DATE(available_at) < DATE('2021-01-01') \
and change_type in ('discount','auto','sale') \
and item_state = 'launched'"

echo CREATE validation dataset table
bq query \
-n 0 \
--destination_table $DATASET_ID.validation \
--replace \
--use_legacy_sql=false \
"select \
     latest_customer_price, \
     latest_promotion_rate, \
     percent_available, \
     latest_base_price, \
     original_price, \
     total, \
     item_views, \
     items_sold, \
     average_retail, \
     designer_id, \
     taxon_permalink, \
     condition, \
     merchandising_category, \
     case when change_type = 'sale' then 1 else 0 end as sold \
from aitest.pricing \
where MOD(ABS(FARM_FINGERPRINT(cast(item_id as STRING))), 100) in (4,5) \
and DATE(available_at) < DATE('2021-05-01') \
and DATE(available_at) > DATE('2021-01-01') \
and change_type in ('discount','auto','sale') \
and item_state = 'launched'"

echo EXTRACT to GCS
bq extract --destination_format CSV $DATASET_ID.training $TRAINING_FILE_PATH
bq extract --destination_format CSV $DATASET_ID.validation $VALIDATION_FILE_PATH