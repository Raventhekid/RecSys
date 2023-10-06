import pandas as pd
import numpy as np

class FeatureHelper:
    @classmethod
    def calc_product_diff(cls, pdf, product_cols, lookback):
        shifted_df = pdf.groupby('customer_code')[product_cols].shift(lookback)
        shifted_df.columns = [f'{col}_shifted' for col in shifted_df.columns]
        shifted_df = shifted_df.fillna(0)
        pdf_grouped = pd.concat([pdf, shifted_df], axis=1)
        for col in product_cols:
            pdf_grouped[f'{col}_diff'] = pdf_grouped[col] - pdf_grouped[f'{col}_shifted']
        return pdf_grouped


    @classmethod
    def calc_new_products(cls, pdf, product_cols):
        diff_cols = [f'{col}_diff' for col in product_cols]
        pdf['num_new_products'] = pdf[diff_cols].apply(lambda row: sum(val > 0 for val in row), axis=1)
        return pdf


    @classmethod
    def calc_dropped_products(cls, pdf, product_cols, lookback):
        shifted_df = pdf.groupby('customer_code')[product_cols].shift(lookback)
        shifted_df = shifted_df.fillna(0)
        shifted_df.columns = [f'{col}_shifted' for col in shifted_df.columns]

        pdf_grouped = pd.concat([pdf, shifted_df], axis=1)

        for col in product_cols:
            pdf_grouped[f'{col}_dropped'] = (pdf_grouped[f'{col}_shifted'] - pdf_grouped[col]).apply(
                lambda x: 1 if x > 0 else 0)

        return pdf_grouped



    @classmethod
    def calc_product_stability(cls, pdf, product_cols, lookback, num_products):
        pdf_grouped = cls.calc_product_diff(pdf, product_cols, lookback)

        stability_cols = [f'{col}_stability' for col in product_cols]
        for col in product_cols:
            pdf_grouped[f'{col}_stability'] = pdf_grouped[f'{col}_diff'].apply(lambda x: 1 if x == 0 else 0)

        # Adding a check for num_products being zero
        num_products = num_products if num_products != 0 else 1

        pdf_grouped['product_stability_score'] = pdf_grouped[stability_cols].sum(axis=1) / num_products
        pdf_grouped = pdf_grouped.fillna(0)

        return pdf_grouped

    @classmethod
    def calc_avg_customer_seniority(cls, pdf, product_cols):
        product_seniority = {}
        for col in product_cols:
            product_seniority[col] = pdf[pdf[col] == 1].groupby('YearMonth')['customer_seniority'].mean()
        return product_seniority


    @classmethod
    def calc_num_customers(cls, pdf, product_cols):
        product_customers = {}
        for col in product_cols:
            product_customers[col] = pdf[pdf[col] == 1].groupby('YearMonth')['customer_code'].nunique()
        return product_customers


    @classmethod
    def calc_avg_age(cls, pdf, product_cols):
        product_avg_age = {}
        for col in product_cols:
            product_avg_age[col] = pdf[pdf[col] == 1].groupby('YearMonth')['age'].mean()
        return product_avg_age