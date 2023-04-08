import pandas as pd
import FileUtils
from HealthCareDataAnalytics import PreProcessing
from HealthCareDataAnalytics import PatternAnalysis
from HealthCareDataAnalytics import PredectiveAnalysis
from HealthCareDataAnalytics import Visualization

pd.set_option('display.max_columns', 20)


# pd.set_option('max_columns', None)


class DataProcessing:

    def analytics(self):
        df = FileUtils.getDataframefromFile()
        df = PreProcessing.dropColumn(df)
        cleaned_df = PreProcessing.imputeValues(df, ["race", "payer_code", "medical_specialty",
                                                     "diag_1", "diag_2", "diag_3",
                                                     "number_diagnoses"])

        # cleaned_df.to_csv("clean_df.csv")

        # Visualization.plot(cleaned_df)

        rules = PatternAnalysis.getPattern(cleaned_df)
        rules.to_csv('all_rules.csv')
        readmittedYesRule = (rules[rules['consequents'] == {'readmitted_Yes'}])
        print("readmittedYesRule=>")
        print(readmittedYesRule.head())

        # ExpiredRule = (rules[rules['consequents'] == {'discharge_disposition_id_Expired'}])
        # print("ExpiredRule=>")
        # print(ExpiredRule.head())

        DiabetesYesRule = (rules[rules['consequents'] == {'diabetesmed_Yes'}])
        print("DiabetesYesRule=>")
        print(DiabetesYesRule.head())

        readmittedYesRule.head(10).to_csv('readmitted_yes_rules.csv')
        # ExpiredRule.head(10).to_csv('expired_rules.csv')
        DiabetesYesRule.head(10).to_csv('diabetes_rules.csv')

        cleaned_df = PredectiveAnalysis.getPrediction(cleaned_df)


if __name__ == '__main__':
    d = DataProcessing()
    d.analytics()

"""
Support < 0.5

"""
