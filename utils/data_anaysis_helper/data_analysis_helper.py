import numpy as np


class DataAnalysis:
    @classmethod
    def replace_missing(cls, pdf, cols, missing_vals=None, **kwargs):
        if missing_vals is None:
            missing_vals = ['NA']
        replace_val = kwargs.get('replace_val', np.nan)
        for col_name in cols:
            pdf[col_name] = pdf[col_name].replace(missing_vals, replace_val)
        return pdf

