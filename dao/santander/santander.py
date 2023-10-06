from settings import SANTANDER_FILE_PATH
import pandas as pd
from pathlib import Path
import numpy as np
from utils.data_anaysis_helper.data_analysis_helper import DataAnalysis
from sklearn.model_selection import train_test_split
from features.feature_engineering import Features
pd.set_option('display.max_columns', None)
from dao.santander.santander_definitions import (column_mapping, cat_missing_values, age_missing_values,
                                                 cust_seniority_missing_values, income_missing_values, cat_cols,
                                                 cat_cols_encode, product_cols, months)




class Santander:
    @classmethod
    def load_data(cls, fname):
        if fname.split('.')[-1] == 'csv':
            pdf = pd.read_csv(Path(SANTANDER_FILE_PATH, fname))
        elif fname.split('.')[-1] == 'xlsx':
            pdf = pd.read_excel(Path(SANTANDER_FILE_PATH, fname))
        else:
            raise Exception("File type must be either CSV or Excel file")

        pdf = pdf.rename(columns=column_mapping)

        pdf['age'] = pdf['age'].astype(str).apply(lambda x: x.strip())
        pdf['customer_seniority'] = pdf['customer_seniority'].astype(str).apply(lambda x: x.strip())
        pdf['household_gross_income'] = pdf['household_gross_income'].astype(str).apply(lambda x: x.strip())

        pdf = DataAnalysis.replace_missing(pdf, cat_cols, missing_vals=cat_missing_values, replace_val=np.nan)
        pdf = DataAnalysis.replace_missing(pdf, ['age'], missing_vals=age_missing_values, replace_val=np.nan)
        pdf = DataAnalysis.replace_missing(pdf, ['customer_seniority'], missing_vals=cust_seniority_missing_values,
                                           replace_val=np.nan)
        pdf = DataAnalysis.replace_missing(pdf, ['household_gross_income'], missing_vals=income_missing_values,
                                           replace_val=np.nan)

        pdf['age'] = pd.to_numeric(pdf['age'], errors='coerce')
        pdf['household_gross_income'] = pd.to_numeric(pdf['household_gross_income'], errors='coerce')
        pdf['customer_seniority'] = pd.to_numeric(pdf['customer_seniority'], errors='coerce')

        pdf.loc[pdf['employee_index'].isnull(), 'employee_index'] = 'N'
        pdf.loc[pdf['country_of_residence'].isnull(), 'country_of_residence'] = 'ES'
        pdf.loc[pdf['sex'].isnull(), 'sex'] = 'V'
        pdf['account_creation_date'] = pdf['account_creation_date'].astype('datetime64[ns]')
        pdf.loc[pdf['account_creation_date'].isnull(), 'account_creation_date'] = pd.Timestamp(2011, 9, 1)
        pdf.loc[pdf['new_customer_index'].isnull(), 'new_customer_index'] = 0
        pdf.loc[pdf['customer_relationship_type'].isnull(), 'customer_relationship_type'] = 1
        pdf['customer_type_at_beginning_of_month'] = pdf['customer_type_at_beginning_of_month'].astype('str').str.slice(
            0, 1)
        pdf.loc[pdf['customer_type_at_beginning_of_month'].isnull(), 'customer_type_at_beginning_of_month'] = '1'
        pdf.loc[pdf['customer_relationship_type_at_beginning_of_month'].isnull(),
        'customer_relationship_type_at_beginning_of_month'] = 'I'
        pdf.loc[pdf['residence_index'].isnull(), 'residence_index'] = 'S'
        pdf.loc[pdf['province_name'].isnull(), 'province_name'] = 'MADRID'
        pdf.loc[pdf['foreigner_index'].isnull(), 'foreigner_index'] = 'N'
        pdf.loc[pdf['channel_used_to_join'].isnull(), 'channel_used_to_join'] = 'MIS'
        pdf.loc[pdf['deceased_index'].isnull(), 'deceased_index'] = 'N'
        pdf.loc[pdf['address_type'].isnull(), 'address_type'] = 0.0
        pdf.loc[pdf['province_code'].isnull(), 'province_code'] = 28.0
        pdf.loc[pdf['activity_index'].isnull(), 'activity_index'] = 0.0
        pdf["household_gross_income"] = pdf[['household_gross_income', 'province_code']].groupby(
            "province_code").transform(lambda x: x.fillna(x.mean()))
        pdf["age"] = pdf[['age', 'province_code']].groupby("province_code").transform(
            lambda x: x.fillna(x.mean()))
        pdf["customer_seniority"] = pdf[['customer_seniority', 'province_code']].groupby("province_code").transform(
            lambda x: x.fillna(x.mean()))
        pdf.loc[pdf['customer_segment'].isnull(), 'customer_segment'] = '02 - PARTICULARES'
        pdf.loc[pdf['payroll'].isnull(), 'payroll'] = 0
        pdf.loc[pdf['pensions_2'].isnull(), 'pensions_2'] = 0

        # pdf['age'] = pd.to_numeric(pdf['age'])
        # pdf['customer_seniority'] = pd.to_numeric(pdf['customer_seniority'])
        # pdf['household_gross_income'] = pd.to_numeric(pdf['household_gross_income'])

        pdf['grass_date'] = pd.to_datetime(pdf['grass_date'], format='%Y-%m-%d')
        pdf['YearMonth'] = pdf['grass_date'].dt.to_period('M')


        #Comment these lines if you want to do analysis
        pdf = pdf.drop('province_code', axis=1)
        pdf = pdf.drop('spouse_index', axis=1)
        pdf = pdf.drop('last_date_as_primary_customer', axis=1)
        pdf = pdf.drop('account_creation_date', axis=1)

        pdf = cls.encoder(pdf)
        return pdf


    @classmethod
    def load_data_dashboard(cls, fname):
        if fname.split('.')[-1] == 'csv':
            pdf = pd.read_csv(Path(SANTANDER_FILE_PATH, fname))
        elif fname.split('.')[-1] == 'xlsx':
            pdf = pd.read_excel(Path(SANTANDER_FILE_PATH, fname))
        else:
            raise Exception("File type must be either CSV or Excel file")

        pdf = pdf.rename(columns=column_mapping)
        return pdf


    @classmethod
    def benchmark(cls, pdf):
        personal_info_cols = [
            'country_of_residence',
            'sex',
            'account_creation_date',
            'last_date_as_primary_customer',
            'residence_index',
            'foreigner_index',
            'channel_used_to_join',
            'address_type',
            'province_name',
            'customer_segment']
        for col in personal_info_cols:
            if col in pdf.columns:
                pdf.drop(col, axis=1, inplace=True)

        return pdf


    @classmethod
    def encoder(cls, pdf):
        for col in cat_cols_encode:
            if col in pdf.columns:
                pdf = pd.concat([pdf, pd.get_dummies(pdf[col], prefix=col)], axis=1)
                pdf.drop(col, axis=1, inplace=True)
        return pdf


    @classmethod
    def sample_customers(cls, pdf, split=0.001, state=2023):

        unique_customer_codes = pdf['customer_code'].unique()

        sampled_customer_codes = pd.Series(unique_customer_codes).sample(frac=split, random_state=state)

        sampled_data = pdf[pdf['customer_code'].isin(sampled_customer_codes)]

        return sampled_data


    @classmethod
    def reformat(cls, pdf, start_year, month, split=0.2, state=2023):
        if month not in months:
            raise Exception(f'Month must be one of the following: {months}')

        train_start_date = pd.to_datetime(start_year + '-' + month + '-' + '28')
        train_end_date = train_start_date + pd.DateOffset(years=1)
        train_label_date = train_end_date + pd.DateOffset(months=1)

        train_mask = (pdf['grass_date'] == train_end_date)
        listed_date = pdf.loc[train_mask].copy()

        train_mask_labels = (pdf['grass_date'] == train_label_date)
        labels = pdf.loc[train_mask_labels].copy()

        listed_date = Features.avg_age(listed_date)
        listed_date = Features.dropped_products(listed_date, 3)
        listed_date = Features.dropped_products(listed_date, 12)
        listed_date = Features.customer_activity(listed_date, 3)
        listed_date = Features.customer_activity(listed_date, 12)
        listed_date = Features.customer_product_stability(listed_date, 3)
        listed_date = Features.customer_product_stability(listed_date, 12)
        listed_date = Features.num_customers(listed_date)
        listed_date = Features.num_prod(listed_date)
        listed_date = Features.cust_sen(listed_date)
        listed_date = Features.avg_customer_seniority(listed_date)
        listed_date = Features.num_prod(listed_date)

        test_end_date = train_label_date
        test_label_date = train_label_date + pd.DateOffset(months=1)

        test_mask = (pdf['grass_date'] == test_end_date)
        test_listed_date = pdf.loc[test_mask].copy()

        test_mask_labels = (pdf['grass_date'] == test_label_date)
        test_labels = pdf.loc[test_mask_labels].copy()

        test_listed_date = Features.avg_age(test_listed_date)
        test_listed_date = Features.dropped_products(test_listed_date, 3)
        test_listed_date = Features.dropped_products(test_listed_date, 12)
        test_listed_date = Features.customer_activity(test_listed_date, 3)
        test_listed_date = Features.customer_activity(test_listed_date, 12)
        test_listed_date = Features.customer_product_stability(test_listed_date, 3)
        test_listed_date = Features.customer_product_stability(test_listed_date, 12)
        test_listed_date = Features.num_customers(test_listed_date)
        test_listed_date = Features.num_prod(test_listed_date)
        test_listed_date = Features.cust_sen(test_listed_date)
        test_listed_date = Features.avg_customer_seniority(test_listed_date)
        test_listed_date = Features.num_prod(test_listed_date)


        for product_col in product_cols:
            listed_date = listed_date.merge(labels[['customer_code', product_col]],
                                            on='customer_code', 
                                            how='left', 
                                            suffixes=('', '_label'))

            listed_date[product_col + '_label'] = listed_date[product_col + '_label'].fillna(0).astype(int)


            test_listed_date = test_listed_date.merge(test_labels[['customer_code', product_col]],
                                            on='customer_code',
                                            how='left',
                                            suffixes=('', '_label'))

            test_listed_date[product_col + '_label'] = test_listed_date[product_col + '_label'].fillna(0).astype(int)

        listed_date_training, listed_date_validation = train_test_split(listed_date, test_size=split,
                                                                        random_state=state)

        return listed_date_training, listed_date_validation, test_listed_date


    @classmethod
    def prepare_data(cls, pdf, start_year, month, split=0.2, state=2023):
        train, validation, test = cls.reformat(pdf, start_year, month, split=split, state=state)
        train = train.drop(columns='grass_date')
        train = train.drop(columns='YearMonth')
        test = test.drop(columns='grass_date')
        test = test.drop(columns='YearMonth')
        validation = validation.drop(columns='grass_date')
        validation = validation.drop(columns='YearMonth')
        return train, validation, test


