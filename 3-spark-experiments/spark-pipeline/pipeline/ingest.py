import logging
import logging.config
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType


class Ingest:
    """ """

    logging.config.fileConfig("config/logging.conf")

    def __init__(self, spark):
        self.spark = spark

    def ingest_data(self):
        """ """

        logger = logging.getLogger("Ingest")

        logger.info("Ingesting from csv")
        PATH_FILE = "data/raw/retailstore.csv"
        customer_df = self.spark.read.csv(PATH_FILE, header=True)
        logger.info("DataFrame created")

        return customer_df
