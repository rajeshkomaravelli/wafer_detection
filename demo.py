from data_validation import Validation
from logger import logging
from data_ingestion import Dbinsertion
from data_preprocessing  import preprocessing
from data_training import Data_training
from predict_data_validation import Pre_Validation
from predict_data_ingestion import predict_Dbinsertion
Validation()
Dbinsertion()
preprocessing()
Data_training()
logging.info("dbinsertion started")




