###########################################################################DATA INGESTION



# working with json
	# json.dump => serializes object to a file
	# json.dumps => transforms object to string 

import json

# Import json
import json

database_config = {
  "host": "10.0.0.5",
  "port": 8456
}

# Open the configuration file in writable mode
with open("database_config.json", "w") as file_handle:
  # Serialize the object in this file handle
  json.dump(obj=database_config, fp=file_handle) 

# Complete the JSON schema
schema = {'properties': {
    'brand': {'type': 'string'},
    'model': {'type': 'string'},
    'price': {'type': 'number'},
    'currency': {'type': 'string'},
    'quantity': {'type': 'integer', 'minimum': 1},
    'date': {'type': 'string', 'format': 'date'}, 
    'countrycode': {'type': 'string', 'pattern': "^[A-Z]{2}$"},
    'store_name': {'type': 'string'}}}

# Write the schema to stdout
singer.write_schema(stream_name='products', schema=schema, key_properties=[])

# Communicate with API
endpoint = "http://localhost:5000"

# Fill in the correct API key
api_key = "scientist007"

# Find out which endpoints are being exposed
api_response = requests.get(f"{endpoint}/{api_key}").json()
pprint.pprint(api_response)

# Show the results of the API's endpoint for the shops
shops = requests.get(f"{endpoint}/{api_key}/diaper/api/v1.0/shops").json()
print(shops)

# Show the items of the shop starting with a "D"
products_of_shop = requests.get(f"{endpoint}/{api_key}/diaper/api/v1.0/items/DM").json()
pprint.pprint(products_of_shop)

# Use the convenience function to query the API
tesco_items = retrieve_products("Tesco")

singer.write_schema(stream_name="products", schema=schema,
                    key_properties=[])

# Write a single record to the stream, that adheres to the schema
singer.write_record(stream_name="products", 
                    record={**tesco_items[0], "store_name": "Tesco"})

for shop in requests.get(SHOPS_URL).json()["shops"]:
    # Write all of the records that you retrieve from the API
    singer.write_records(
      stream_name="products", # Use the same stream name that you used in the schema
      records=({**item, "store_name": shop}
               for item in retrieve_products(shop))
    )



###########################################################################SPARK DEPLOYMENT




# Creating a deployable artifact
repl:~/workspace/spark_pipelines$ zip --recurse-paths pydiaper.zip pydiaper
# Submitting a job
repl:~/workspace$ spark-submit --py-files spark_pipelines/pydiaper/pydiaper.zip spark_pipelines/pydiaper/pydiaper/cleaning/clean_ratings.py 



###########################################################################UNIT TESTING



# unit testing

from datetime import date
from pyspark.sql import Row

Record = Row("country", "utm_campaign", "airtime_in_minutes", "start_date", "end_date")

# Create a tuple of records
data = (
  Record("USA", "DiapersFirst", 28, date(2017, 1, 20), date(2017, 1, 27)),
  Record("Germany", "WindelKind", 31, date(2017, 1, 25), None),
  Record("India", "CloseToCloth", 32, date(2017, 1, 25), date(2017, 2, 2))
)

# Create a DataFrame from these records
frame = spark.createDataFrame(data)
frame.show()

from .chinese_provinces_improved import \
    aggregate_inhabitants_by_province
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, \
    StructField, StringType, LongType, BooleanType


def test_aggregate_inhabitants_by_province():
    """The number of inhabitants per province should be aggregated,
    regardless of their distinctive features.
    """

    spark = SparkSession.builder.getOrCreate()

    fields = [
        StructField("country", StringType(), True),
        StructField("province", StringType(), True),
        StructField("inhabitants", LongType(), True),
        StructField("foo", BooleanType(), True),  # distinctive features
    ]

    frame = spark.createDataFrame({
        ("China", "A", 3, False),
        ("China", "A", 2, True),
        ("China", "B", 14, False),
        ("US", "A", 4, False)},
        schema=StructType(fields)
    )
    actual = aggregate_inhabitants_by_province(frame).cache()

    # In the older implementation, the data was first filtered for a specific
    # country, after which you'd aggregate by province. The same province
    # name could occur in multiple countries though.
    # This test is expecting the data to be grouped by country,
    # then province from aggregate_inhabitants_by_province()
    expected = spark.createDataFrame(
        {("China", "A", 5), ("China", "B", 14), ("US", "A", 4)},
        schema=StructType(fields[:3])
    ).cache()

    assert actual.schema == expected.schema, "schemas don't match up"
    assert sorted(actual.collect()) == sorted(expected.collect()),\
        "data isn't equal"


 from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, sum

from .catalog import catalog


def extract_demographics(sparksession, catalog):
    return sparksession.read.parquet(catalog["clean/demographics"])


def store_chinese_demographics(frame, catalog):
    frame.write.parquet(catalog["business/chinese_demographics"])


# Improved aggregation function, grouped by country and province
def aggregate_inhabitants_by_province(frame):
    return (frame
            .groupBy("province")
            .agg(sum(col("inhabitants")).alias("inhabitants"))
            )


def main():
    spark = SparkSession.builder.getOrCreate()
    frame = extract_demographics(spark, catalog)
    chinese_demographics = frame.filter(lower(col("country")) == "china")
    aggregated_demographics = aggregate_inhabitants_by_province(
        chinese_demographics
    )
    store_chinese_demographics(aggregated_demographics, catalog)


if __name__ == "__main__":
    main()



###########################################################################AIRFLOW



from datetime import datetime
from airflow import DAG

reporting_dag = DAG(
    dag_id="publish_EMEA_sales_report", 
    # Insert the cron expression
    schedule_interval="0 7 * * 1",
    start_date=datetime(2019, 11, 24),
    default_args={"owner": "sales"}
)

"""
┌─── minute (0 - 59)
│ ┌─── hour (0 - 23)
│ │ ┌─── day of the month (1 - 31)
│ │ │ ┌─── month (1 - 12)
│ │ │ │ ┌─── day of the week (0 - 6) (Sunday to Saturday;
│ │ │ │ │               7 is also Sunday on some systems)
* * * * *
"""

# cron dependencies
# Specify direction using verbose method
prepare_crust.set_downstream(apply_tomato_sauce)

tasks_with_tomato_sauce_parent = [add_cheese, add_ham, add_olives, add_mushroom]
for task in tasks_with_tomato_sauce_parent:
    # Specify direction using verbose method on relevant task
    apply_tomato_sauce.set_downstream(task)

# Specify direction using bitshift operator
tasks_with_tomato_sauce_parent >> bake_pizza

# Specify direction using verbose method
bake_pizza.set_upstream(prepare_oven)

# dag for daily pipelines
# Create a DAG object
dag = DAG(
  dag_id='optimize_diaper_purchases',
  default_args={
    # Don't email on failure
    'email_on_failure': False,
    # Specify when tasks should have started earliest
    'start_date': datetime(2019, 6, 25)
  },
  # Run the DAG daily
  schedule_interval='@daily')

# schedule bash scripts via airflow
config = os.path.join(os.environ["AIRFLOW_HOME"], 
                      "scripts",
                      "configs", 
                      "data_lake.conf")

ingest = BashOperator(
  # Assign a descriptive id
  task_id="ingest_data", 
  # Complete the ingestion pipeline
  bash_command='tap-marketing-api | target-csv --config %s' % config,
  dag=dag)

# schedule spark jobs via airflow
# Import the operator
from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator

# Set the path for our files.
entry_point = os.path.join(os.environ["AIRFLOW_HOME"], "scripts", "clean_ratings.py")
dependency_path = os.path.join(os.environ["AIRFLOW_HOME"], "dependencies", "pydiaper.zip")

with DAG('data_pipeline', start_date=datetime(2019, 6, 25),
         schedule_interval='@daily') as dag:
    # Define task clean, running a cleaning job.
    clean_data = SparkSubmitOperator(
        application=entry_point, 
        py_files=dependency_path,
        task_id='clean_data',
        conn_id='spark_default')

# deploy pipeline
spark_args = {"py_files": dependency_path,
              "conn_id": "spark_default"}
# Define ingest, clean and transform job.
with dag:
    ingest = BashOperator(task_id='Ingest_data', bash_command='tap-marketing-api | target-csv --config %s' % config)
    clean = SparkSubmitOperator(application=clean_path, task_id='clean_data', **spark_args)
    insight = SparkSubmitOperator(application=transform_path, task_id='show_report', **spark_args)
    
    # set triggering sequence
    ingest >> clean >> insight



###########################################################################