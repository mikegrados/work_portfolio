import sys
from pipeline import (
    ingest,
    transform,
    persist,
)
import logging
import logging.config
import pyspark
from pyspark.sql import SparkSession


class Pipeline:
    """ """

    logging.config.fileConfig("config/logging.conf")

    def create_spark_session(self):
        """ """
        self.spark = (
            SparkSession.builder.appName("First spark app")
            .enableHiveSupport()
            .getOrCreate()
        )

    def run_pipeline(self):
        """ """
        try:
            logging.info("run_pipeline method started")
            ingest_process = ingest.Ingest(self.spark)
            df = ingest_process.ingest_data()

            transform_process = transform.Transform(self.spark)
            transformed_df = transform_process.transform_data(df)
            transformed_df

            persist_process = persist.Persist(self.spark)
            persist_process.persist_data(transformed_df)

            logging.info("run_pipeline method ended")

        except Exception as exp:
            logging.error(f"Error running the pipeline >: {str(exp)}")
            sys.exit(1)

        return None


if __name__ == "__main__":

    pipeline = Pipeline()
    logging.info("Application started")

    pipeline.create_spark_session()
    logging.info("Spark session created")

    pipeline.run_pipeline()
    logging.info("Pipeline executed")
