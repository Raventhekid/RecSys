from dao.santander.santander import Santander
from models.model import CustomerSimilarity
from datetime import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
import json



def train_and_monitor(data, start_year, start_month, threshold, retrain=1, benchmark=0):
    current_date = datetime(int(start_year), int(start_month), 1)
    end_date = pd.to_datetime(data['grass_date']).max()

    model = None
    logs = {}

    while current_date < end_date:
        print(f"Processing window: {current_date.strftime('%Y-%m')}")

        test_label_date = current_date + relativedelta(years=1, months=2)

        if test_label_date > end_date:
            print(
                f"Final data available up to {end_date.strftime('%Y-%m')}. Can't proceed beyond {test_label_date.strftime('%Y-%m')}.")
            break

        if benchmark==1:
            data = Santander.benchmark(data)
        sample = Santander.sample_customers(data)
        year, month = current_date.strftime('%Y'), current_date.strftime('%m')
        final_train, final_val, final_test = Santander.prepare_data(sample, year, month)

        X_train, y_train = CustomerSimilarity.reformat_all(final_train)
        X_test, y_test = CustomerSimilarity.reformat_all(final_test)
        # X_val, y_val = CustomerSimilarity.reformat_all(final_val)

        if model is None:
            if benchmark==1:
                model = CustomerSimilarity.train_logistic_regression_model(X_train, y_train)
            else:
                model = CustomerSimilarity.train_ranking_model(X_train, y_train)

        newprod_test = CustomerSimilarity.customers_with_new_products(final_test)
        recall_test = CustomerSimilarity.average_recall(model, X_test, y_test, newprod_test)
        print("Recall test:", recall_test)

        window_end = current_date + relativedelta(months=1)
        print(f"Window end: {window_end.strftime('%Y-%m')}")

        next_year_same_month = current_date + relativedelta(years=1)
        window_key = f"{current_date.strftime('%Y-%m')} to {next_year_same_month.strftime('%Y-%m')}"

        logs[window_key] = recall_test

        if retrain == 1 and recall_test < threshold:
            model = CustomerSimilarity.train_ranking_model(X_train, y_train)
            recall_test_after_retrain = CustomerSimilarity.average_recall(model, X_test, y_test, newprod_test)

            if recall_test_after_retrain < threshold:
                raise Exception("Human intervention required")

        current_date = current_date + relativedelta(months=1)

    return logs


data = Santander.load_data('train_ver2.csv')
start_year = '2015'
start_month = '01'
threshold = 0.3

logs = train_and_monitor(data, start_year, start_month, threshold)

with open('logs_2.json', 'w') as file:
    json.dump(logs, file)