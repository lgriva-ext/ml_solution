"""data drift monitoring
"""
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import os
import pandas as pd
import numpy as np
import datetime
import sys

from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset, RegressionPreset
from evidently.metrics import *

from evidently.test_suite import TestSuite
from evidently.tests.base_test import generate_column_tests
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset, RegressionTestPreset
from evidently.tests import *


def compute_drift(reference_df, current_df):
    vars_to_ignore = ["index"]

    report = Report(metrics=[
        DataDriftPreset(),
    ])

    report.run(reference_data=reference_df[[c for c in reference_df.columns if c not in vars_to_ignore]], current_data=current_df[[c for c in current_df.columns if c not in vars_to_ignore]])

    report_dict = report.as_dict()
    for d in report_dict["metrics"]:
        if d["metric"] == "DataDriftTable":
            dd = d["result"]["drift_by_columns"]

    drift = sum([dd[c]["drift_detected"] for c in dd]) >= 5

    return drift


def get_data(limit_train_date, start_date=None):
    driver = "com.microsoft.sqlserver.jdbc.SQLServerDriver"

    database_host = "lugrserver.database.windows.net"
    database_port = "1433" # update if you use a non-default port
    database_name = "lgdb"
    table = "hist_data"
    user = "lguser"
    # THIS SHOULD BE IN GITHUB AND SEND AS AN ARGUMENT
    password = "Passdbmy0"

    #url = f"jdbc:sqlserver://{database_host}:{database_port}/{database_name}"
    url = f"jdbc:sqlserver://{database_host}:{database_port};database={database_name}"
    #;user=lguser@lugrserver;password={your_password_here};encrypt=true;trustServerCertificate=false;hostNameInCertificate=*.database.windows.net;loginTimeout=30;"

    remote_table_sdf = (spark.read
    .format("jdbc")
    .option("driver", driver)
    .option("url", url)
    .option("dbtable", table)
    .option("user", user)
    .option("password", password)
    .load()
    )
    remote_table_pdf = remote_table_sdf.toPandas()

    if not start_date:
       start_date = "1970-05-05"

    train_data = remote_table_pdf[(
       remote_table_pdf["date_created"]<=limit_train_date)&(
       remote_table_pdf["date_created"]<=start_date)].copy()
    new_data = remote_table_pdf[
        remote_table_pdf["date_created"]>limit_train_date].copy()

    return train_data, new_data


if __name__ == "__main__":
    train_date = sys.argv[1] #os.environ["train_date"]
    df_train_data, df_new_data = get_data(
        #start_date=datetime.datetime.now(),
        limit_train_date=datetime.datetime.strftime(train_date, "%Y-%m-%d"))

    drift_detected = compute_drift(df_train_data, df_new_data)

    if drift_detected:
        sys.exit(1)
