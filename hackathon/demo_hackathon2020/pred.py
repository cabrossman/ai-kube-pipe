import base64
import os

from google.protobuf import text_format
from googleapiclient import discovery

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow.python.lib.io import file_io
from tensorflow_transform import coders as tft_coders
from tensorflow_transform.tf_metadata import schema_utils, dataset_schema
from tfx.utils import io_utils


def _get_raw_feature_spec(schema):
    return schema_utils.schema_as_feature_spec(schema).feature_spec


def _make_proto_coder(schema):
    raw_feature_spec = _get_raw_feature_spec(schema)
    raw_schema = dataset_schema.from_feature_spec(raw_feature_spec)
    return tft_coders.ExampleProtoCoder(raw_schema)


def _make_csv_coder(schema, column_names):
    raw_feature_spec = _get_raw_feature_spec(schema)
    parsing_schema = dataset_schema.from_feature_spec(raw_feature_spec)
    return tft_coders.CsvCoder(column_names, parsing_schema)


def _read_schema(path):
    result = schema_pb2.Schema()
    contents = file_io.read_file_to_string(path)
    text_format.Parse(contents, result)
    return result


class CMLEPredictor:
    """Interface for CMLE Predictor"""

    def __init__(self, name, version):
        self.name = name
        self.version = version
        self.project_id = 'trr-assignments-staging' # os.environ.get('GCP_PROJECT_ID')

        self.client = discovery.build('ml', 'v1', cache_discovery=False)
        self.model = self.client.projects()

        self.schema_path = self._get_schema_path()
        self.schema = _read_schema(self.schema_path)

    def _get_schema_path(self):
        version = self.model.models().versions().get(
            name='projects/{}/models/{}/versions/{}'.format(
                self.project_id,
                self.name,
                self.version
            )
        ).execute()
        model_path = version['deploymentUri']
        pipe_path = f"gs://{os.path.join(*model_path[5:].split('/')[:-4])}"
        # Is it always `107`?
        return os.path.join(pipe_path, 'SchemaGen', 'schema', '107', 'schema.pbtxt')

    def encode_input(self, path):
        column_names = io_utils.load_csv_column_names(path)
        proto_coder = _make_proto_coder(self.schema)
        csv_coder = _make_csv_coder(self.schema, column_names)

        input_file = open(path, 'r')
        input_file.readline()  # skip header line
        lines = input_file.readlines()

        encoded = []
        for i, l in enumerate(lines):
            csv_l = csv_coder.decode(l)
            proto_l = proto_coder.encode(csv_l)
            encoded.append({"examples": { "b64": base64.b64encode(proto_l).decode('utf-8')}})
        return encoded

    def predict(self, path):
        """Runs inference using Cloud ML Engine"""

        res = self.model.predict(
            name='projects/{}/models/{}/versions/{}'.format(
                self.project_id, self.name, self.version
            ),
            body={'instances': self.encode_input(path)}
        ).execute()

        if 'error' in res:
            raise RuntimeError(res['error'])

        return res['predictions']


def run_predictions(path):
    model = CMLEPredictor(
        name='pipeline_hackathon2020',
        version='vserving_model_dir'
    )
    preds = model.predict(path)
    print('Predictions:', preds)


if __name__ == '__main__':
    run_predictions('data/pred.csv')