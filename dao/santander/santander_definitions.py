import numpy as np

column_mapping = {'fecha_dato': 'grass_date',
                  'ncodpers': 'customer_code',
                  'ind_empleado': 'employee_index',
                  'pais_residencia': 'country_of_residence',
                  'sexo': 'sex',
                  'age': 'age',
                  'fecha_alta': 'account_creation_date',
                  'ind_nuevo': 'new_customer_index',
                  'antiguedad': 'customer_seniority',
                  'indrel': 'customer_relationship_type',
                  'ult_fec_cli_1t': 'last_date_as_primary_customer',
                  'indrel_1mes': 'customer_type_at_beginning_of_month',
                  'tiprel_1mes': 'customer_relationship_type_at_beginning_of_month',
                  'indresi': 'residence_index',
                  'indext': 'foreigner_index',
                  'conyuemp': 'spouse_index',
                  'canal_entrada': 'channel_used_to_join',
                  'indfall': 'deceased_index',
                  'tipodom': 'address_type',
                  'cod_prov': 'province_code',
                  'nomprov': 'province_name',
                  'ind_actividad_cliente': 'activity_index',
                  'renta': 'household_gross_income',
                  'segmento': 'customer_segment',
                  'ind_ahor_fin_ult1': 'savings_account',
                  'ind_aval_fin_ult1': 'guarantees',
                  'ind_cco_fin_ult1': 'current_account',
                  'ind_cder_fin_ult1': 'derivatives_account',
                  'ind_cno_fin_ult1': 'payroll_account',
                  'ind_ctju_fin_ult1': 'junior_account',
                  'ind_ctma_fin_ult1': 'más_particular_account',
                  'ind_ctop_fin_ult1': 'particular_account',
                  'ind_ctpp_fin_ult1': 'particular_plus_account',
                  'ind_deco_fin_ult1': 'short-term_deposits',
                  'ind_deme_fin_ult1': 'medium-term_deposits',
                  'ind_dela_fin_ult1': 'long-term_deposits',
                  'ind_ecue_fin_ult1': 'e-account',
                  'ind_fond_fin_ult1': 'funds',
                  'ind_hip_fin_ult1': 'mortgage',
                  'ind_plan_fin_ult1': 'pensions',
                  'ind_pres_fin_ult1': 'loans',
                  'ind_reca_fin_ult1': 'taxes',
                  'ind_tjcr_fin_ult1': 'credit_card',
                  'ind_valo_fin_ult1': 'securities',
                  'ind_viv_fin_ult1': 'home_account',
                  'ind_nomina_ult1': 'payroll',
                  'ind_nom_pens_ult1': 'pensions_2',
                  'ind_recibo_ult1': 'direct_debit'}


cat_missing_values = ['NA', 'Na', 'nA', 'na', ' NA', ' Na', ' na', ' nA', 'NA ', 'Na ', 'na ', 'nA ', np.nan, ' ',
                      '', '     NA', 'nan']


age_missing_values = ['NA', 'Na', 'nA', 'na', 'nan', 'naN', 'nAn', 'nAN', 'Nan', 'NaN', 'NAn', 'NAN', 'Nan', np.nan,
                      '-1']


cust_seniority_missing_values = ['-999999', 'NA', 'Na', 'nA', 'na', 'nan', 'naN', 'nAn', 'nAN', 'Nan', 'NaN', 'NAn',
                                 'NAN', 'Nan', np.nan]


income_missing_values = ['-999999', 'NA', 'Na', 'nA', 'na', 'nan', 'naN', 'nAn', 'nAN', 'Nan', 'NaN', 'NAn',
                         'NAN', 'Nan', np.nan]


cat_cols_analysis = ['employee_index', 'country_of_residence', 'sex',
                     'new_customer_index', 'customer_relationship_type',
                     'customer_type_at_beginning_of_month',
                     'customer_relationship_type_at_beginning_of_month', 'residence_index',
                     'foreigner_index', 'channel_used_to_join', 'deceased_index',
                     'address_type', 'province_name', 'activity_index',
                     'customer_segment', 'savings_account', 'guarantees', 'current_account',
                     'derivatives_account', 'payroll_account', 'junior_account',
                     'más_particular_account', 'particular_account', 'particular_plus_account',
                     'short-term_deposits', 'medium-term_deposits', 'long-term_deposits',
                     'e-account', 'funds', 'mortgage', 'loans', 'taxes', 'credit_card',
                     'securities', 'home_account', 'payroll', 'direct_debit', 'pensions', 'pensions_2']


cat_cols = ['employee_index', 'account_creation_date', 'country_of_residence', 'sex',
            'new_customer_index', 'customer_relationship_type',
            'customer_type_at_beginning_of_month',
            'customer_relationship_type_at_beginning_of_month', 'residence_index',
            'foreigner_index', 'channel_used_to_join', 'deceased_index',
            'address_type', 'province_name', 'activity_index',
            'customer_segment', 'savings_account', 'guarantees', 'current_account',
            'derivatives_account', 'payroll_account', 'junior_account',
            'más_particular_account', 'particular_account', 'particular_plus_account',
            'short-term_deposits', 'medium-term_deposits', 'long-term_deposits',
            'e-account', 'funds', 'mortgage', 'loans', 'taxes', 'credit_card',
            'securities', 'home_account', 'payroll', 'direct_debit', 'pensions', 'pensions_2']


original_columns = ['savings_account', 'guarantees', 'current_account',
                    'derivatives_account', 'payroll_account', 'junior_account',
                    'más_particular_account', 'particular_account', 'particular_plus_account',
                    'short-term_deposits', 'medium-term_deposits', 'long-term_deposits',
                    'e-account', 'funds', 'mortgage', 'loans', 'taxes', 'credit_card',
                    'securities', 'home_account', 'payroll', 'direct_debit',
                    'YearMonth', 'customer_code', 'grass_date',
                    'employee_index', 'country_of_residence', 'sex', 'age',
                    'account_creation_date', 'new_customer_index', 'customer_seniority',
                    'customer_relationship_type', 'customer_type_at_beginning_of_month',
                    'customer_relationship_type_at_beginning_of_month', 'residence_index',
                    'foreigner_index', 'channel_used_to_join', 'deceased_index', 'address_type',
                    'province_name', 'activity_index', 'household_gross_income', 'customer_segment',
                    'next_month_products']


cat_cols_encode = ['employee_index', 'country_of_residence', 'sex',
                   'customer_relationship_type_at_beginning_of_month',
                   'customer_type_at_beginning_of_month',
                   'residence_index', 'foreigner_index', 'channel_used_to_join',
                   'deceased_index', 'customer_segment', 'province_name']


product_cols = ['savings_account', 'guarantees', 'current_account',
                'derivatives_account', 'payroll_account', 'junior_account',
                'más_particular_account', 'particular_account', 'particular_plus_account',
                'short-term_deposits', 'medium-term_deposits', 'long-term_deposits',
                'e-account', 'funds', 'mortgage', 'loans', 'taxes', 'credit_card',
                'securities', 'home_account', 'payroll', 'direct_debit', 'pensions', 'pensions_2']


label_cols = [f'{product}_label' for product in product_cols]


num_cols = ['age', 'customer_seniority', 'household_gross_income']


months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

