import logging
import logging.config
import pyspark
from pyspark.sql import SparkSession


class Transform:
    """ """

    logging.config.fileConfig("config/logging.conf")

    def __init__(self, spark):

        self.spark = spark

    def transform_data(self, df):
        """

        Parameters
        ----------
        df :
            

        Returns
        -------

        """

        logger = logging.getLogger("Transform")
        logger.info("Transforming")
        df1 = df.na.drop()
        return df1
