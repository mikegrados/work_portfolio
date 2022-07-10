import logging
import logging.config
import pyspark
from pyspark.sql import SparkSession


class Persist:
    """ """

    logging.config.fileConfig("config/logging.conf")

    def __init__(self, spark):
        self.spark = spark

    def persist_data(self, df):  # sourcery skip: raise-specific-error
        """

        Parameters
        ----------
        df :
            

        Returns
        -------

        """
        try:
            logger = logging.getLogger("Persist")
            logger.info("Persisting")
            df.coalesce(1).write.option("header", "true").csv(
                "data/transformed_retailstore"
            )
        except Exception as exp:
            logger.error(f"An error ocurred while persisting data > {str(exp)}")
            raise Exception("HDFS directory already exist") from exp
