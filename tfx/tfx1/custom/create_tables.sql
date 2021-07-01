# Supply a project!


CREATE TABLE `$PROJECT.trrsample.l2i_train` as 
SELECT *, 
    JSON_EXTRACT(store_distances, '$[0].closest_location') AS closest_location,
    JSON_EXTRACT(store_distances, '$[0].closest_location_distance') AS closest_location_distance
FROM `$PROJECT.trrsample.l2i`
WHERE MOD(ABS(FARM_FINGERPRINT(cast(user_id as String))), 10) <= 7
AND date < TIMESTAMP('2021-04-01')

CREATE TABLE `$PROJECT.trrsample.l2i_train_dev` as 
SELECT *
FROM `$PROJECT.trrsample.l2i_train`
WHERE RAND() < 0.01


CREATE TABLE `$PROJECT.trrsample.l2i_eval` as 
SELECT *, 
    JSON_EXTRACT(store_distances, '$[0].closest_location') AS closest_location,
    JSON_EXTRACT(store_distances, '$[0].closest_location_distance') AS closest_location_distance
FROM `$PROJECT.trrsample.l2i`
WHERE MOD(ABS(FARM_FINGERPRINT(cast(user_id as String))), 10) = 8
AND date BETWEEN TIMESTAMP('2021-04-01') AND TIMESTAMP('2021-04-30')

CREATE TABLE `$PROJECT.trrsample.l2i_eval_dev` as 
SELECT *
FROM `$PROJECT.trrsample.l2i_eval`
WHERE RAND() < 0.01


CREATE TABLE `$PROJECT.trrsample.l2i_test` as 
SELECT *, 
    JSON_EXTRACT(store_distances, '$[0].closest_location') AS closest_location,
    JSON_EXTRACT(store_distances, '$[0].closest_location_distance') AS closest_location_distance
FROM `$PROJECT.trrsample.l2i`
WHERE MOD(ABS(FARM_FINGERPRINT(cast(user_id as String))), 10) = 9
AND date BETWEEN TIMESTAMP('2021-05-01') AND TIMESTAMP('2021-05-15')