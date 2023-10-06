import pandas as pd
import numpy as np
from utils.feature_engineering_helper.feature_engineering_helper import FeatureHelper
from dao.santander.santander_definitions import product_cols


class Features:
    # product_cols = ['savings_account', 'guarantees', 'current_account',
    # 'derivatives_account', 'payroll_account', 'junior_account',
    # 'más_particular_account', 'particular_account', 'particular_plus_account',
    # 'short-term_deposits', 'medium-term_deposits', 'long-term_deposits',
    # 'e-account', 'funds', 'mortgage', 'loans', 'taxes', 'credit_card',
    # 'securities', 'home_account', 'payroll', 'direct_debit']


    @classmethod
    def num_prod(cls, pdf):
        end_date = pdf['grass_date'].max()
        start_date = pdf['grass_date'].min()
        mask = (pdf['grass_date'] >= start_date) & (pdf['grass_date'] <= end_date)
        window = pdf.loc[mask].copy()
        window['product_count'] = 0
        for column in product_cols:
            window['product_count'] += window[column]
        group = window.groupby(['YearMonth', 'customer_code'])['product_count'].sum().reset_index()
        pdf = pd.merge(pdf, group, how='left', on=['YearMonth', 'customer_code'])
        pdf = pdf.fillna(0)
        return pdf


    @classmethod
    def cust_sen(cls, pdf):
        pdf['product_count'] = pdf['product_count'].fillna(0)
        pdf['customer_seniority'] = pdf['customer_seniority'].fillna(0)

        pdf.loc[pdf['customer_seniority'] == 0, 'customer_seniority'] = 1

        pdf['products_divided_seniority'] = pdf['product_count'] / pdf['customer_seniority']

        pdf['products_divided_seniority'] = pdf['products_divided_seniority'].replace([np.inf, -np.inf], np.nan)
        pdf['products_divided_seniority'] = pdf['products_divided_seniority'].fillna(0)

        return pdf


    @classmethod
    def customer_activity(cls, pdf, lookback):
        pdf_copy = pdf.copy()
        pdf_grouped = pdf_copy.groupby(['customer_code', 'YearMonth'])[product_cols].sum().reset_index()
        pdf_grouped.sort_values(['customer_code', 'YearMonth'], inplace=True)

        pdf_grouped = FeatureHelper.calc_product_diff(pdf_grouped, product_cols, lookback)
        pdf_grouped = pdf_grouped.fillna(0)
        pdf_grouped = FeatureHelper.calc_new_products(pdf_grouped, product_cols)

        activity_column_name = f'customer_activity_in_past_{lookback}_months'
        pdf_grouped[activity_column_name] = pdf_grouped.groupby('customer_code')['num_new_products'].rolling(
            lookback).sum().reset_index(0, drop=True)
        pdf_grouped[activity_column_name] = pdf_grouped[activity_column_name].fillna(0)

        pdf = pd.merge(pdf, pdf_grouped[['customer_code', 'YearMonth', activity_column_name]],
                       on=['customer_code', 'YearMonth'], how='left')
        pdf = pdf.fillna(0)
        return pdf


    @classmethod
    def dropped_products(cls, pdf, lookback):
        pdf_copy = pdf.copy()
        pdf_grouped = pdf_copy.groupby(['customer_code', 'YearMonth'])[product_cols].sum().reset_index()
        pdf_grouped.sort_values(['customer_code', 'YearMonth'], inplace=True)

        pdf_grouped = FeatureHelper.calc_dropped_products(pdf_grouped, product_cols, lookback)

        dropped_cols = [f'{col}_dropped' for col in product_cols]
        pdf_grouped['num_dropped_products'] = pdf_grouped[dropped_cols].sum(axis=1)

        new_column_name = f'num_dropped_products_{lookback}_months'
        pdf_grouped.rename(columns={'num_dropped_products': new_column_name}, inplace=True)

        pdf = pd.merge(pdf, pdf_grouped[['customer_code', 'YearMonth', new_column_name]],
                       on=['customer_code', 'YearMonth'], how='left')
        pdf = pdf.fillna(0)
        return pdf


    @classmethod
    def customer_product_stability(cls, pdf, lookback):
        pdf_copy = pdf.copy()
        pdf_grouped = pdf_copy.groupby(['customer_code', 'YearMonth'])[product_cols].sum().reset_index()
        pdf_grouped.sort_values(['customer_code', 'YearMonth'], inplace=True)

        num_products = len(product_cols)
        pdf_grouped = FeatureHelper.calc_product_stability(pdf_grouped, product_cols, lookback, num_products)

        new_column_name = f'product_stability_score_ratio_{lookback}_months'
        pdf_grouped.rename(columns={'product_stability_score': new_column_name}, inplace=True)

        pdf = pd.merge(pdf, pdf_grouped[['customer_code', 'YearMonth', new_column_name]],
                       on=['customer_code', 'YearMonth'], how='left')
        pdf = pdf.fillna(0)
        return pdf


    @classmethod
    def avg_customer_seniority(cls, pdf):
        avg_seniority = FeatureHelper.calc_avg_customer_seniority(pdf, product_cols)
        for product, avg_sen in avg_seniority.items():
            pdf = pdf.merge(avg_sen.reset_index().rename(columns={'customer_seniority': f'avg_seniority_{product}'}),
                            on='YearMonth', how='left')
            pdf[f'avg_seniority_{product}'] = pdf[f'avg_seniority_{product}'].fillna(0)
        return pdf


    @classmethod
    def num_customers(cls, pdf):
        num_customers = FeatureHelper.calc_num_customers(pdf, product_cols)
        for product, num_cust in num_customers.items():
            pdf = pdf.merge(num_cust.reset_index().rename(columns={'customer_code': f'num_customers_{product}'}),
                            on='YearMonth', how='left')
        pdf = pdf.fillna(0)
        return pdf


    @classmethod
    def avg_age(cls, pdf):
        avg_age = FeatureHelper.calc_avg_age(pdf, product_cols)
        for product, avg_age_product in avg_age.items():
            pdf = pdf.merge(avg_age_product.reset_index().rename(columns={'age': f'avg_age_{product}'}),
                            on='YearMonth', how='left')
        pdf = pdf.fillna(0)
        return pdf