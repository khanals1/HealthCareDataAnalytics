import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def getPattern(df):
    categorical_columns = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id",
                           "admission_source_id", "time_in_hospital","payer_code","medical_specialty", "num_lab_procedures",
                           "num_procedures", "num_medications", "number_outpatient",
                           "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
                           #"metformin","glipizide","glyburide","pioglitazone","rosiglitazone", "insulin","change",
                           "diabetesmed", "readmitted"]
    df = pd.get_dummies(df, columns=categorical_columns)
    freqItems = apriori(df, min_support=0.2, use_colnames=True, max_len=4, verbose=1)
    rules = association_rules(freqItems, metric="confidence", min_threshold=0.5)
    rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

    return rules
