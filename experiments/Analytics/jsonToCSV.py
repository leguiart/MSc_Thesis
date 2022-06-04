
import sys
import pandas as pd
import json
import os

from common.Analytics import QD_Analytics
from common.Constants import *
# sys.path.append(os.getcwd() + "/..")

def jsonToCSV(filename):
    if (not "MNSLC" in filename) and (not "QN" in filename):
        experiment = "SO"
        analytics_csv = ANALYTICS_JSON_SO.replace(".json", ".csv")
    
    elif "QN" in filename:
        experiment = "QN-MOEA"
        analytics_csv = ANALYTICS_JSON_QN.replace(".json", ".csv")

    elif "MNSLC" in filename:
        experiment = "MNSLC"
        analytics_csv = ANALYTICS_JSON_MNSLC.replace(".json", ".csv")
    with open(filename, 'r') as fh:
        for line in fh:
            analytics = json.loads(line)
            keys = list(analytics.keys())
            run = keys[0]
            qd_history = analytics[run]
            qd_analytics = QD_Analytics(int(run), experiment)
            qd_analytics.qd_history = qd_history
            df = qd_analytics.to_dataframe()
            df.to_csv(analytics_csv, mode='a', header=not os.path.exists(analytics_csv), index = False)

def main(argv):
    jsonToCSV(argv[0])

if __name__ == "__main__":
    main(sys.argv[1:])