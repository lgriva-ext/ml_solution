"""_summary_
"""
from databricks import feature_store


feature_store_uri = f"databricks://featurestore:featurestore"
fs = feature_store.FeatureStoreClient(feature_store_uri=feature_store_uri)


def extract_data():
    pass


if __name__ == "__main__":

    df = extract_data()
    df_test_schema = df.schema
    try:
        fs.write_table(name="train_data_raw", df=df, mode="merge")
    except Exception as e:
        print(e)
        fs.create_table(
            name="train_data_raw",
            primary_keys=["item_id"],
            schema=df_test_schema,
            description="raw test bigmart features",
        )
        fs.write_table(name="train_data_raw", df=df, mode="merge")
