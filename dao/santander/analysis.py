import pandas as pd
import numpy as np
from dao.santander.santander_definitions import (cat_cols_analysis, original_columns, product_cols, label_cols,
                                                 num_cols)



class Analysis:
    @classmethod
    def analyze_data(cls, pdf):
        num_analysis_pdf = pd.DataFrame(columns=['Column Name', 'Missing Rate', 'Zero Rate', 'Mean', 'Variance', '20%',
                                                 '40%', '60%', '80%', 'Min Value', 'Max Value'])

        for i, col in enumerate(num_cols):
            missing_rate = pdf[col].isnull().mean() * 100
            zero_rate = (pdf[col] == 0).mean() * 100
            mean = np.mean(pdf[col])
            variance = np.var(pdf[col])
            quantiles = np.nanpercentile(pdf[col], [20, 40, 60, 80])
            min_val = np.min(pdf[col])
            max_val = np.max(pdf[col])

            num_analysis_pdf.loc[i, 'Column Name'] = col
            num_analysis_pdf.loc[i, 'Missing Rate'] = missing_rate
            num_analysis_pdf.loc[i, 'Zero Rate'] = zero_rate
            num_analysis_pdf.loc[i, 'Mean'] = mean
            num_analysis_pdf.loc[i, 'Variance'] = variance
            num_analysis_pdf.loc[i, '20%'] = quantiles[0]
            num_analysis_pdf.loc[i, '40%'] = quantiles[1]
            num_analysis_pdf.loc[i, '60%'] = quantiles[2]
            num_analysis_pdf.loc[i, '80%'] = quantiles[3]
            num_analysis_pdf.loc[i, 'Min Value'] = min_val
            num_analysis_pdf.loc[i, 'Max Value'] = max_val

        print("Numerical Analysis")
        print(num_analysis_pdf)

        cat_analysis_pdf = pd.DataFrame(columns=['Column Name', 'Missing Rate', 'Most Popular'])
        for i, col in enumerate(cat_cols_analysis):
            missing_rate = pdf[col].isnull().mean() * 100
            most_popular = pdf[col].value_counts().nlargest(5).index.tolist() if not pdf[col].mode().empty else np.nan

            cat_analysis_pdf.loc[i, 'Column Name'] = col
            cat_analysis_pdf.loc[i, 'Missing Rate'] = missing_rate
            cat_analysis_pdf.loc[i, 'Most Popular'] = most_popular

        print("Categorical Analysis")
        print(cat_analysis_pdf)

        unique_customers_per_month = pdf.groupby('YearMonth')['customer_code'].nunique()
        unique_customers_per_month = unique_customers_per_month.reset_index()
        unique_customers_per_month.columns = ['Month', 'Unique Customers']

        print(unique_customers_per_month)

        monthly_customers = pdf.groupby('YearMonth')['customer_code'].unique()
        num_clients = pd.DataFrame(columns=['YearMonth', 'new_customers', 'lost_customers'])
        for i in range(1, len(monthly_customers)):
            prev_month_customers = set(monthly_customers[i - 1])
            curr_month_customers = set(monthly_customers[i])

            new_customers = len(curr_month_customers - prev_month_customers)
            lost_customers = len(prev_month_customers - curr_month_customers)

            num_clients.loc[i] = [monthly_customers.index[i], new_customers, lost_customers]

        print('New and Lost Clients per Month')
        print(num_clients)

        monthly_assets = pdf.groupby('YearMonth')[product_cols].sum()

        print('Assets per Month')
        print(monthly_assets)


@classmethod
def count_new_lost(cls, curr_month, prev_month):
    new_customers = curr_month - prev_month
    lost_customers = prev_month - curr_month if prev_month > curr_month else 0
    return new_customers, lost_customers


@classmethod
def final_form(cls, pdf):
    label_cols = [col for col in pdf.columns if '_label' in col]

    pdf['next_month_products'] = pdf[label_cols].apply(lambda row: dict(row), axis=1)

    pdf = pdf.drop(columns=label_cols)
    pdf = pdf.drop(columns='grass_date')
    pdf = pdf.drop(columns='YearMonth')

    print(pdf)
    return pdf


@classmethod
def statistics(cls, train_pdf, val_pdf, test_pdf):
    stats = {}

    stats['total_rows_train'] = train_pdf.shape[0]
    stats['total_rows_val'] = val_pdf.shape[0]
    stats['total_rows_test'] = test_pdf.shape[0]

    stats['unique_customers_train'] = train_pdf['customer_code'].nunique()
    stats['unique_customers_val'] = val_pdf['customer_code'].nunique()
    stats['unique_customers_test'] = test_pdf['customer_code'].nunique()

    stats['duplicated_rows_train'] = train_pdf.loc[:, train_pdf.columns != 'next_month_products'].duplicated().sum()
    stats['duplicated_rows_val'] = val_pdf.loc[:, val_pdf.columns != 'next_month_products'].duplicated().sum()
    stats['duplicated_rows_test'] = test_pdf.loc[:, test_pdf.columns != 'next_month_products'].duplicated().sum()

    stats['duplicated_customers_train'] = train_pdf.duplicated(subset=['customer_code']).sum()
    stats['duplicated_customers_val'] = val_pdf.duplicated(subset=['customer_code']).sum()
    stats['duplicated_customers_test'] = test_pdf.duplicated(subset=['customer_code']).sum()

    train_val_customers = set(train_pdf['customer_code']).union(val_pdf['customer_code'])

    common_customers_train_val_test = train_val_customers.intersection(test_pdf['customer_code'])

    stats['common_customers_train_val_test'] = len(common_customers_train_val_test)

    print(stats)
    return stats


@classmethod
def label_statistic(cls, pdf):
    inner = pdf.copy()

    for product, label in zip(product_cols, label_cols):
        inner[product + '_change'] = inner[product] != inner[label]

    product_change_counts = inner[[product + '_change' for product in product_cols]].sum(axis=1)
    num_customers_changed = (product_change_counts > 0).sum()

    print(num_customers_changed)
    return num_customers_changed


@classmethod
def final_form_analysis(cls, pdf):
    engineered_features = [col for col in pdf.columns if col not in original_columns]

    num_analysis_pdf = pd.DataFrame(index=engineered_features,
                                    columns=['Missing Rate', 'Zero Rate', 'Mean', 'Variance', '20%', '40%', '60%',
                                             '80%', 'Min Value', 'Max Value'])

    for i, col in enumerate(engineered_features):
        missing_rate = pdf[col].isnull().mean() * 100
        zero_rate = (pdf[col] == 0).mean() * 100
        mean = np.mean(pdf[col])
        variance = np.var(pdf[col])
        quantiles = np.nanpercentile(pdf[col].dropna(), [20, 40, 60, 80])
        min_val = np.min(pdf[col])
        max_val = np.max(pdf[col])

        num_analysis_pdf.loc[col] = [missing_rate, zero_rate, mean, variance, *quantiles, min_val, max_val]

    print("Numerical Analysis")
    print(num_analysis_pdf)