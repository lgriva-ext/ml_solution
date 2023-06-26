import json
import os
import datetime
from datetime import date
import pandas as pd
import logging
from databricks import feature_store
import sys
from pyspark.sql.functions import col
from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset, RegressionTestPreset
from evidently.tests import *


def get_feature_store():
    feature_store_uri = f"databricks://featurestore:featurestore"
    fs = feature_store.FeatureStoreClient(feature_store_uri=feature_store_uri)
    return fs


def execute():
    fs = get_feature_store()
    cut_date = sys.argv[1]

    df = fs.read_table("train_data_preprocessed")#.toPandas()
    reference = df.where((col('date_feat') <= cut_date)).toPandas()
    current = df.where(col('date_feat') > cut_date).toPandas()

    report = Report(metrics=[
        DataDriftPreset(),
    ])

    report.run(reference_data=reference, current_data=current)

    ### DECIDE IF RETRAIN OR NOT AND PRINT CERTAIN OUTPUT ###


if __name__ == "__main__":
    execute()
