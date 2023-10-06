import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity
from dao.santander.santander import Santander


class RecommendationSystem:
    usecols = ['customer_code', 'age', 'new_customer_index', 'customer_seniority',
       'customer_relationship_type', 'address_type', 'activity_index',
       'household_gross_income', 'savings_account', 'guarantees',
       'current_account', 'derivatives_account', 'payroll_account',
       'junior_account', 'más_particular_account', 'particular_account',
       'particular_plus_account', 'short-term_deposits',
       'medium-term_deposits', 'long-term_deposits', 'e-account', 'funds',
       'mortgage', 'pensions', 'loans', 'taxes', 'credit_card',
       'securities', 'home_account', 'payroll', 'pensions_2',
       'direct_debit', 'YearMonth', 'avg_age_savings_account',
       'avg_age_guarantees', 'avg_age_current_account',
       'avg_age_derivatives_account', 'avg_age_payroll_account',
       'avg_age_junior_account', 'avg_age_más_particular_account',
       'avg_age_particular_account', 'avg_age_particular_plus_account',
       'avg_age_short-term_deposits', 'avg_age_medium-term_deposits',
       'avg_age_long-term_deposits', 'avg_age_e-account', 'avg_age_funds',
       'avg_age_mortgage', 'avg_age_loans', 'avg_age_taxes',
       'avg_age_credit_card', 'avg_age_securities',
       'avg_age_home_account', 'avg_age_payroll', 'avg_age_direct_debit',
       'num_dropped_products_3_months', 'num_dropped_products_12_months',
       'customer_activity_in_past_3_months',
       'customer_activity_in_past_12_months',
       'product_stability_score_ratio_3_months',
       'product_stability_score_ratio_12_months',
       'num_customers_savings_account', 'num_customers_guarantees',
       'num_customers_current_account',
       'num_customers_derivatives_account',
       'num_customers_payroll_account', 'num_customers_junior_account',
       'num_customers_más_particular_account',
       'num_customers_particular_account',
       'num_customers_particular_plus_account',
       'num_customers_short-term_deposits',
       'num_customers_medium-term_deposits',
       'num_customers_long-term_deposits', 'num_customers_e-account',
       'num_customers_funds', 'num_customers_mortgage',
       'num_customers_loans', 'num_customers_taxes',
       'num_customers_credit_card', 'num_customers_securities',
       'num_customers_home_account', 'num_customers_payroll',
       'num_customers_direct_debit', 'product_count_x',
       'products_divided_seniority', 'avg_seniority_savings_account',
       'avg_seniority_guarantees', 'avg_seniority_current_account',
       'avg_seniority_derivatives_account',
       'avg_seniority_payroll_account', 'avg_seniority_junior_account',
       'avg_seniority_más_particular_account',
       'avg_seniority_particular_account',
       'avg_seniority_particular_plus_account',
       'avg_seniority_short-term_deposits',
       'avg_seniority_medium-term_deposits',
       'avg_seniority_long-term_deposits', 'avg_seniority_e-account',
       'avg_seniority_funds', 'avg_seniority_mortgage',
       'avg_seniority_loans', 'avg_seniority_taxes',
       'avg_seniority_credit_card', 'avg_seniority_securities',
       'avg_seniority_home_account', 'avg_seniority_payroll',
       'avg_seniority_direct_debit', 'product_count_y', 'employee_index_A',
       'employee_index_B', 'employee_index_F', 'employee_index_N',
       'country_of_residence_AD', 'country_of_residence_AE',
       'country_of_residence_AO', 'country_of_residence_AR',
       'country_of_residence_AT', 'country_of_residence_AU',
       'country_of_residence_BE', 'country_of_residence_BG',
       'country_of_residence_BO', 'country_of_residence_BR',
       'country_of_residence_BY', 'country_of_residence_BZ',
       'country_of_residence_CA', 'country_of_residence_CH',
       'country_of_residence_CL', 'country_of_residence_CM',
       'country_of_residence_CO', 'country_of_residence_CU',
       'country_of_residence_DE', 'country_of_residence_DK',
       'country_of_residence_DO', 'country_of_residence_EC',
       'country_of_residence_ES', 'country_of_residence_FI',
       'country_of_residence_FR', 'country_of_residence_GB',
       'country_of_residence_GH', 'country_of_residence_GN',
       'country_of_residence_GQ', 'country_of_residence_GR',
       'country_of_residence_HN', 'country_of_residence_IE',
       'country_of_residence_IL', 'country_of_residence_IN',
       'country_of_residence_IT', 'country_of_residence_KE',
       'country_of_residence_KR', 'country_of_residence_KW',
       'country_of_residence_LT', 'country_of_residence_LU',
       'country_of_residence_MA', 'country_of_residence_MD',
       'country_of_residence_ML', 'country_of_residence_MX',
       'country_of_residence_NL', 'country_of_residence_PE',
       'country_of_residence_PK', 'country_of_residence_PT',
       'country_of_residence_PY', 'country_of_residence_QA',
       'country_of_residence_RO', 'country_of_residence_RU',
       'country_of_residence_SE', 'country_of_residence_SG',
       'country_of_residence_US', 'country_of_residence_UY',
       'country_of_residence_VE', 'country_of_residence_ZA', 'sex_H',
       'sex_V', 'customer_relationship_type_at_beginning_of_month_A',
       'customer_relationship_type_at_beginning_of_month_I',
       'customer_relationship_type_at_beginning_of_month_P',
       'customer_relationship_type_at_beginning_of_month_R',
       'customer_type_at_beginning_of_month_1',
       'customer_type_at_beginning_of_month_2',
       'customer_type_at_beginning_of_month_3',
       'customer_type_at_beginning_of_month_4',
       'customer_type_at_beginning_of_month_P',
       'customer_type_at_beginning_of_month_n', 'residence_index_N',
       'residence_index_S', 'foreigner_index_N', 'foreigner_index_S',
       'channel_used_to_join_007', 'channel_used_to_join_013',
       'channel_used_to_join_KAA', 'channel_used_to_join_KAB',
       'channel_used_to_join_KAC', 'channel_used_to_join_KAD',
       'channel_used_to_join_KAE', 'channel_used_to_join_KAF',
       'channel_used_to_join_KAG', 'channel_used_to_join_KAH',
       'channel_used_to_join_KAI', 'channel_used_to_join_KAJ',
       'channel_used_to_join_KAK', 'channel_used_to_join_KAL',
       'channel_used_to_join_KAM', 'channel_used_to_join_KAN',
       'channel_used_to_join_KAO', 'channel_used_to_join_KAP',
       'channel_used_to_join_KAQ', 'channel_used_to_join_KAR',
       'channel_used_to_join_KAS', 'channel_used_to_join_KAT',
       'channel_used_to_join_KAU', 'channel_used_to_join_KAW',
       'channel_used_to_join_KAY', 'channel_used_to_join_KAZ',
       'channel_used_to_join_KBB', 'channel_used_to_join_KBD',
       'channel_used_to_join_KBE', 'channel_used_to_join_KBF',
       'channel_used_to_join_KBG', 'channel_used_to_join_KBH',
       'channel_used_to_join_KBJ', 'channel_used_to_join_KBL',
       'channel_used_to_join_KBM', 'channel_used_to_join_KBO',
       'channel_used_to_join_KBQ', 'channel_used_to_join_KBR',
       'channel_used_to_join_KBS', 'channel_used_to_join_KBU',
       'channel_used_to_join_KBV', 'channel_used_to_join_KBW',
       'channel_used_to_join_KBX', 'channel_used_to_join_KBY',
       'channel_used_to_join_KBZ', 'channel_used_to_join_KCA',
       'channel_used_to_join_KCB', 'channel_used_to_join_KCC',
       'channel_used_to_join_KCD', 'channel_used_to_join_KCE',
       'channel_used_to_join_KCF', 'channel_used_to_join_KCG',
       'channel_used_to_join_KCH', 'channel_used_to_join_KCI',
       'channel_used_to_join_KCJ', 'channel_used_to_join_KCK',
       'channel_used_to_join_KCL', 'channel_used_to_join_KCM',
       'channel_used_to_join_KCN', 'channel_used_to_join_KCO',
       'channel_used_to_join_KCQ', 'channel_used_to_join_KCS',
       'channel_used_to_join_KCT', 'channel_used_to_join_KCU',
       'channel_used_to_join_KCV', 'channel_used_to_join_KDA',
       'channel_used_to_join_KDC', 'channel_used_to_join_KDD',
       'channel_used_to_join_KDE', 'channel_used_to_join_KDG',
       'channel_used_to_join_KDH', 'channel_used_to_join_KDM',
       'channel_used_to_join_KDN', 'channel_used_to_join_KDO',
       'channel_used_to_join_KDP', 'channel_used_to_join_KDQ',
       'channel_used_to_join_KDR', 'channel_used_to_join_KDS',
       'channel_used_to_join_KDT', 'channel_used_to_join_KDU',
       'channel_used_to_join_KDV', 'channel_used_to_join_KDW',
       'channel_used_to_join_KDX', 'channel_used_to_join_KDY',
       'channel_used_to_join_KDZ', 'channel_used_to_join_KEA',
       'channel_used_to_join_KEB', 'channel_used_to_join_KEC',
       'channel_used_to_join_KED', 'channel_used_to_join_KEE',
       'channel_used_to_join_KEF', 'channel_used_to_join_KEG',
       'channel_used_to_join_KEH', 'channel_used_to_join_KEI',
       'channel_used_to_join_KEJ', 'channel_used_to_join_KEK',
       'channel_used_to_join_KEL', 'channel_used_to_join_KEN',
       'channel_used_to_join_KEO', 'channel_used_to_join_KEQ',
       'channel_used_to_join_KES', 'channel_used_to_join_KEU',
       'channel_used_to_join_KEV', 'channel_used_to_join_KEW',
       'channel_used_to_join_KEY', 'channel_used_to_join_KEZ',
       'channel_used_to_join_KFA', 'channel_used_to_join_KFC',
       'channel_used_to_join_KFD', 'channel_used_to_join_KFE',
       'channel_used_to_join_KFF', 'channel_used_to_join_KFG',
       'channel_used_to_join_KFH', 'channel_used_to_join_KFI',
       'channel_used_to_join_KFJ', 'channel_used_to_join_KFK',
       'channel_used_to_join_KFL', 'channel_used_to_join_KFM',
       'channel_used_to_join_KFN', 'channel_used_to_join_KFP',
       'channel_used_to_join_KFS', 'channel_used_to_join_KFT',
       'channel_used_to_join_KFU', 'channel_used_to_join_KGC',
       'channel_used_to_join_KGV', 'channel_used_to_join_KGW',
       'channel_used_to_join_KGX', 'channel_used_to_join_KGY',
       'channel_used_to_join_KHC', 'channel_used_to_join_KHD',
       'channel_used_to_join_KHE', 'channel_used_to_join_KHF',
       'channel_used_to_join_KHK', 'channel_used_to_join_KHL',
       'channel_used_to_join_KHM', 'channel_used_to_join_KHN',
       'channel_used_to_join_KHO', 'channel_used_to_join_KHP',
       'channel_used_to_join_KHQ', 'channel_used_to_join_MIS',
       'channel_used_to_join_RED', 'deceased_index_N', 'deceased_index_S',
       'customer_segment_01 - TOP', 'customer_segment_02 - PARTICULARES',
       'customer_segment_03 - UNIVERSITARIO', 'province_name_ALAVA',
       'province_name_ALBACETE', 'province_name_ALICANTE',
       'province_name_ALMERIA', 'province_name_ASTURIAS',
       'province_name_AVILA', 'province_name_BADAJOZ',
       'province_name_BALEARS, ILLES', 'province_name_BARCELONA',
       'province_name_BIZKAIA', 'province_name_BURGOS',
       'province_name_CACERES', 'province_name_CADIZ',
       'province_name_CANTABRIA', 'province_name_CASTELLON',
       'province_name_CEUTA', 'province_name_CIUDAD REAL',
       'province_name_CORDOBA', 'province_name_CORUÑA, A',
       'province_name_CUENCA', 'province_name_GIPUZKOA',
       'province_name_GIRONA', 'province_name_GRANADA',
       'province_name_GUADALAJARA', 'province_name_HUELVA',
       'province_name_HUESCA', 'province_name_JAEN', 'province_name_LEON',
       'province_name_LERIDA', 'province_name_LUGO',
       'province_name_MADRID', 'province_name_MALAGA',
       'province_name_MELILLA', 'province_name_MURCIA',
       'province_name_NAVARRA', 'province_name_OURENSE',
       'province_name_PALENCIA', 'province_name_PALMAS, LAS',
       'province_name_PONTEVEDRA', 'province_name_RIOJA, LA',
       'province_name_SALAMANCA', 'province_name_SANTA CRUZ DE TENERIFE',
       'province_name_SEGOVIA', 'province_name_SEVILLA',
       'province_name_SORIA', 'province_name_TARRAGONA',
       'province_name_TERUEL', 'province_name_TOLEDO',
       'province_name_VALENCIA', 'province_name_VALLADOLID',
       'province_name_ZAMORA', 'province_name_ZARAGOZA']


    @classmethod
    def data_preprocessing(cls, pdf):
        if isinstance(pdf['next_month_products'].iloc[0], str):
            pdf['next_month_products'] = pdf['next_month_products'].apply(ast.literal_eval)

        product_columns = [key for key in pdf['next_month_products'].iloc[0]]
        for col in product_columns:
            pdf[col] = pdf['next_month_products'].apply(lambda x: x[col])

        pdf.drop(columns=['next_month_products'], inplace=True)
        print(pdf)
        return pdf

    @classmethod
    def generate_candidates(cls, data, customer_code, top_n):
     user_feature_matrix = data[cls.usecols]

     # Get the row corresponding to the input customer_code
     target_user_row = user_feature_matrix[data['customer_code'] == customer_code]

     user_similarity = cosine_similarity(target_user_row, user_feature_matrix)
     user_similarity_series = pd.Series(user_similarity.flatten(), index=data['customer_code'])

     top_similar_users = user_similarity_series.nlargest(top_n + 1).iloc[1:]

     # Get the data for the top similar users
     similar_users_data = data[data['customer_code'].isin(top_similar_users.index)]

     similar_users_products = similar_users_data[Santander.product_cols]

     target_user_products = data[data['customer_code'] == customer_code][Santander.product_cols]
     candidate_products = similar_users_products.loc[:, ~(similar_users_products & target_user_products).any()]

     return candidate_products


# preprocessed = RecommendationSystem.data_preprocessing(final_train)
candidates = RecommendationSystem.generate_candidates(preprocessed, 1074483, 50)