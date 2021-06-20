"""
Big Query Split Examles - https://www.tensorflow.org/tfx/guide/examplegen

### Example Gen - split locally
output_config = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(splits=[ 
        example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=4),
        example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
    ]))
example_gen = tfx.components.CsvExampleGen(
    input_base=DATA_ROOT,
    output_config=output_config)
context.run(example_gen)



Example with pre split dataset
QUERY_TEMPLATE = 'SELECT * FROM trr-assignments-staging.covertype_dataset.{}'
input = example_gen_pb2.Input(splits=[
                example_gen_pb2.Input.Split(name='train', pattern=QUERY_TEMPLATE.format('training')),
                example_gen_pb2.Input.Split(name='eval', pattern=QUERY_TEMPLATE.format('validation'))
            ])
example_gen = BigQueryExampleGen(input_config=input)
context.run(example_gen, beam_pipeline_args=BEAM_PIPELINE_ARGS)


Example with partition field
output_config = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(splits=[ 
        example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=4),
        example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
    ],
    partition_feature_name='Soil_Type'))
example_gen = BigQueryExampleGen(query=QUERY_TEMPLATE.format('covertype'),output_config=output_config)
context.run(example_gen, beam_pipeline_args=BEAM_PIPELINE_ARGS) ###RUN

### Write out examples
examples_uri = example_gen.outputs['examples'].get()[0].uri
tfrecord_filenames = [os.path.join(examples_uri, 'train', name)
                      for name in os.listdir(os.path.join(examples_uri, 'train'))]
dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")
for tfrecord in dataset.take(2):
  example = tf.train.Example()
  example.ParseFromString(tfrecord.numpy())
  for name, feature in example.features.feature.items():
    if feature.HasField('bytes_list'):
        value = feature.bytes_list.value
    if feature.HasField('float_list'):
        value = feature.float_list.value
    if feature.HasField('int64_list'):
        value = feature.int64_list.value
    print('{}: {}'.format(name, value))
  print('******')
"""


def make_schema(context, example_gen, SCHEMA_DIR, SCHEMA_FILE):
    # Good Detail Here : https://www.tensorflow.org/tfx/tutorials/data_validation/tfdv_basic
    # Guide : https://github.com/tensorflow/tfx/blob/master/docs/guide/tfdv.md
    from tfx.components import StatisticsGen, SchemaGen
    import tensorflow_data_validation as tfdv
    from tensorflow_metadata.proto.v0 import schema_pb2
    from tensorflow.io.gfile import makedirs

    ### Statistics Gen
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples'])
    context.run(statistics_gen) ###RUN
    # context.show(statistics_gen.outputs['statistics']) #This shows the HTML generated data graphs

    ### Schema Gen - I guess you normally dont do this one? 
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=False)
    context.run(schema_gen) ###RUN
    #context.show(schema_gen.outputs['schema']) # shows table of data, type, presence, valency, domain

    schema_proto_path = '{}/{}'.format(schema_gen.outputs['schema'].get()[0].uri, 'schema.pbtxt')
    schema = tfdv.load_schema_text(schema_proto_path)

    tfdv.set_domain(schema, 'Cover_Type', schema_pb2.IntDomain(name='Cover_Type', min=0, max=6, is_categorical=True))
    tfdv.set_domain(schema, 'Slope',  schema_pb2.IntDomain(name='Slope', min=0, max=90))
    ### tfdv.display_schema(schema=schema) shows new schema, has domain in some cases

    ### Write the Schema
    makedirs(SCHEMA_DIR)
    tfdv.write_schema_text(schema, SCHEMA_FILE)

    ###!cat {schema_file} show the schema file



"""
TRANSFORM NOTES
#os.listdir(transform.outputs['transform_graph'].get()[0].uri)
#os.listdir(transform.outputs['transformed_examples'].get()[0].uri)

transform_uri = transform.outputs['transformed_examples'].get()[0].uri
tfrecord_filenames = [os.path.join(transform_uri,  'train', name)
                      for name in os.listdir(os.path.join(transform_uri, 'train'))]
dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")
for tfrecord in dataset.take(2):
  example = tf.train.Example()
  example.ParseFromString(tfrecord.numpy())
  for name, feature in example.features.feature.items():
    if feature.HasField('bytes_list'):
        value = feature.bytes_list.value
    if feature.HasField('float_list'):
        value = feature.float_list.value
    if feature.HasField('int64_list'):
        value = feature.int64_list.value
    print('{}: {}'.format(name, value))
  print('******')
"""


""" Training view in TensorBoard
logs_path = trainer.outputs['model_run'].get()[0].uri
print(logs_path)
Open a new JupyterLab terminal window
tensorboard dev upload --logdir [YOUR_LOGDIR]
"""

def hypertune(context, transform, schema_importer, TRAINER_MODULE_FILE):
    from tfx.components import Tuner, ImporterNode, Trainer
    from tfx.proto import trainer_pb2
    from tfx.types.standard_artifacts import HyperParameters
    from tfx.dsl.components.base import executor_spec
    from tfx.components.trainer import executor as trainer_executor
    ### Tuner - Hyperparams
    tuner = Tuner(
            module_file=TRAINER_MODULE_FILE,
            examples=transform.outputs['transformed_examples'],
            transform_graph=transform.outputs['transform_graph'],
            train_args=trainer_pb2.TrainArgs(num_steps=1000),
            eval_args=trainer_pb2.EvalArgs(num_steps=500))
    context.run(tuner) ###RUN!

    ### Now get best resuslts
    hparams_importer = ImporterNode(
        instance_name='import_hparams',
        # This can be Tuner's output file or manually edited file. The file contains
        # text format of hyperparameters (kerastuner.HyperParameters.get_config())
        source_uri=tuner.outputs.best_hyperparameters.get()[0].uri,
        artifact_type=HyperParameters)
    context.run(hparams_importer) ###RUN1

    ### Now actualy retrain with those values
    trainer = Trainer(
        custom_executor_spec=executor_spec.ExecutorClassSpec(trainer_executor.GenericExecutor),
        module_file=TRAINER_MODULE_FILE,
        transformed_examples=transform.outputs.transformed_examples,
        schema=schema_importer.outputs.result,
        transform_graph=transform.outputs.transform_graph,
        hyperparameters=hparams_importer.outputs.result,    
        train_args=trainer_pb2.TrainArgs(splits=['train'], num_steps=5000),
        eval_args=trainer_pb2.EvalArgs(splits=['eval'], num_steps=1000))
    context.run(trainer) ### RUN!

    return trainer



""" MODEL ANALYZER
# TODO: Your code here to create a tfma.MetricThreshold. 
# Review the API documentation here: https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/MetricThreshold
# Hint: Review the API documentation for tfma.GenericValueThreshold to constrain accuracy between 50% and 99%.

model_blessing_uri = model_analyzer.outputs.blessing.get()[0].uri
#!ls -l {model_blessing_uri}
evaluation_uri = model_analyzer.outputs['evaluation'].get()[0].uri
#!ls {evaluation_uri}
eval_result = tfma.load_eval_result(evaluation_uri)

eval_result
tfma.view.render_slicing_metrics(eval_result)
tfma.view.render_slicing_metrics(
    eval_result, slicing_column='Wilderness_Area')
"""


""" INFRA VALIDATOR
infra_blessing_uri = infra_validator.outputs.blessing.get()[0].uri
#!ls -l {infra_blessing_uri}
"""


""" PURSHER STUFF
pusher.outputs
latest_pushed_model = os.path.join(SERVING_MODEL_DIR, max(os.listdir(SERVING_MODEL_DIR)))
!saved_model_cli show --dir {latest_pushed_model} --all
"""