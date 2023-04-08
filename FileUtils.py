import pandas as pd
import psycopg2

from sqlalchemy import create_engine, text


def getDataframefromFile():
    alchemyEngine  = create_engine('postgresql://shirish@127.0.0.1:5432/patientdata')
    dbConnection = alchemyEngine.connect()
    #file = "10kPatients.csv"
    getdata = text("SELECT * FROM updated_patient_data")
    df = pd.read_sql(getdata, dbConnection)
    return df
