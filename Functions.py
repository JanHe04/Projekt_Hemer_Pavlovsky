def import_libraries():
    try:
        import requests
        globals()['requests'] = requests

        from bs4 import BeautifulSoup
        globals()['BeautifulSoup'] = BeautifulSoup

        import pandas as pd
        globals()['pd'] = pd

        import sklearn
        globals()['sklearn'] = sklearn


        import statsmodels.api as sm
        globals()['sm'] = sm

        from sklearn.model_selection import train_test_split
        globals()['train_test_split'] = train_test_split

        from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
        globals()['confusion_matrix'] = confusion_matrix
        globals()['classification_report'] = classification_report
        globals()['roc_curve'] = roc_curve
        globals()['auc'] = auc

        import matplotlib.pyplot as plt
        globals()['plt'] = plt

        import numpy as np
        globals()['np'] = np

        import lightgbm as lgb
        globals()['lgb'] = lgb

        print("All libraries are imported successfully.")

    except ImportError as e:
        library_name = str(e).split("No module named ")[-1].replace("'", "")
        print(f"Could not import the library {library_name}. Please ensure it is installed.")
