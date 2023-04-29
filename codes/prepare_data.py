import os
from utils import get_path, get_spark


def read_raw_data(spark, path):
    """
    The read_raw_data function reads in the raw data from the SEER_1975_2016_CUSTOM_TEXTDATA folder.
    It then creates a temporary view called RAW that can be used to query and manipulate the data.

    Parameters
    ----------
        spark
            Create a sparksession object
        path
            Specify the location of the data files

    Returns
    -------

        A dataframe with the column name &quot;raw_data&quot;

    Doc Author
    ----------
        Trelent
    """
    files = path["datasets"] / "SEER_1975_2016_CUSTOM_TEXTDATA"
    df = spark.read.csv(
        str(files), pathGlobFilter="*.TXT", recursiveFileLookup=True, header=False
    )
    df = df.withColumnRenamed("_c0", "RAW_DATA")
    df.createOrReplaceTempView("RAW")
    return


def preprocess_data(spark, path):
    """
    The preprocess_data function performs the following steps:
        1. Reads the queries from repository and creates a Spark DataFrame
        2. Performs feature engineering on the data to create new features that will be used for modeling
        3. Saves the table in memory for faster processing

    Parameters
    ----------
        spark
            Create a sparksession object
        path
            Load the queries from the query folder

    Returns
    -------

        Nothing

    Doc Author
    ----------
        Trelent
    """
    for name in ["pre_processing_1", "pre_processing_2", "pre_processing_3"]:
        query = path["queries"] / f"{name}.sql"
        spark.sql(query.read_text()).createOrReplaceTempView(name)

    spark.sql("CACHE TABLE pre_processing_3 OPTIONS ('storageLevel' = 'MEMORY_ONLY_SER')")
    return


def save_processed_data(spark, path):
    """
    The save_processed_data function takes a SparkSession object and a pathlib.Path object as arguments,
    and saves the pre_processing_3 table to the processed subdirectory of datasets in parquet format.
    The function then renames the file to processed.snappy.parquet.

    Parameters
    ----------
        spark
            Access the spark session
        path
            Specify the path where the processed data will be saved

    Returns
    -------

        The path to the processed data

    Doc Author
    ----------
        Trelent
    """
    to_save = path["datasets"] / "processed"
    df = spark.table("pre_processing_3").coalesce(1)
    df.write.parquet(str(to_save), mode="overwrite")
    filename = list(to_save.rglob("*.parquet"))[0]
    os.rename(filename, to_save / "processed.snappy.parquet")
    return

if __name__ == "__main__":
    path = get_path()
    spark = get_spark()

    read_raw_data(spark, path)
    preprocess_data(spark, path)
    save_processed_data(spark, path)
