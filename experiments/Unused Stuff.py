#Santander.py line 100 or so. right after stripping name, seniority and income
# avg_income_by_province = pdf[pdf['household_gross_income'] >= 1].groupby('province_name')[
#     'household_gross_income'].mean()
# pdf.loc[pdf['household_gross_income'] < 1, 'household_gross_income'] = pdf['province_name'].map(
#     avg_income_by_province)
# pdf['household_gross_income'].fillna(pdf['household_gross_income'].mean(), inplace=True)


# @classmethod
# def get_sample(cls, pdf, start_year, month, split=0.2, state=2023):
#     if month not in cls.months:
#         raise Exception(f'Month must be one of the following: {cls.months}')
#
#     start_date = pd.to_datetime(start_year + '-' + month)
#     end_date = start_date + pd.DateOffset(years=1, months=1)
#
#     mask = (pdf['grass_date'] >= start_date) & (pdf['grass_date'] < end_date)
#     window = pdf.loc[mask].copy()
#
#     cust_id = window['customer_code'].unique()
#     train_codes, val_codes = train_test_split(cust_id, test_size=split, random_state=state)
#     mask_train = window['customer_code'].isin(train_codes)
#     mask_val = window['customer_code'].isin(val_codes)
#
#     train = window.loc[mask_train].copy()
#     validation = window.loc[mask_val].copy()
#
#     test_start_date = start_date + pd.DateOffset(months=1)
#     test_end_date = test_start_date + pd.DateOffset(years=1, months=1)
#
#     test_mask = (pdf['grass_date'] >= test_start_date) & (pdf['grass_date'] < test_end_date)
#     test = pdf.loc[test_mask].copy()
#
#     for col in cls.product_cols:
#         new_col_name = col + '_label'
#         train[new_col_name] = train.groupby('customer_code')[col].shift(-1)
#         train[new_col_name].fillna(0, inplace=True)
#
#         validation[new_col_name] = validation.groupby('customer_code')[col].shift(-1)
#         validation[new_col_name].fillna(0, inplace=True)
#
#     train = train[train['grass_date'] < test_start_date]
#     validation = validation[validation['grass_date'] < test_start_date]
#
#     print('Training Data')
#     print(train)
#     print('Validation Data')
#     print(validation)
#     print('Testing Data')
#     print(test)
#
#     return train, validation, test



#SANTANDER METHODS AND TESTS
# @classmethod
# def analyze_data(cls, pdf):
#     num_analysis_pdf = pd.DataFrame(columns=['Column Name', 'Missing Rate', 'Zero Rate', 'Mean', 'Variance', '20%',
#                                              '40%', '60%', '80%', 'Min Value', 'Max Value'])
#
#     for i, col in enumerate(cls.num_cols):
#         missing_rate = pdf[col].isnull().mean() * 100
#         zero_rate = (pdf[col] == 0).mean() * 100
#         mean = np.mean(pdf[col])
#         variance = np.var(pdf[col])
#         quantiles = np.nanpercentile(pdf[col], [20, 40, 60, 80])
#         min_val = np.min(pdf[col])
#         max_val = np.max(pdf[col])
#
#         num_analysis_pdf.loc[i, 'Column Name'] = col
#         num_analysis_pdf.loc[i, 'Missing Rate'] = missing_rate
#         num_analysis_pdf.loc[i, 'Zero Rate'] = zero_rate
#         num_analysis_pdf.loc[i, 'Mean'] = mean
#         num_analysis_pdf.loc[i, 'Variance'] = variance
#         num_analysis_pdf.loc[i, '20%'] = quantiles[0]
#         num_analysis_pdf.loc[i, '40%'] = quantiles[1]
#         num_analysis_pdf.loc[i, '60%'] = quantiles[2]
#         num_analysis_pdf.loc[i, '80%'] = quantiles[3]
#         num_analysis_pdf.loc[i, 'Min Value'] = min_val
#         num_analysis_pdf.loc[i, 'Max Value'] = max_val
#
#     print("Numerical Analysis")
#     print(num_analysis_pdf)
#
#     cat_analysis_pdf = pd.DataFrame(columns=['Column Name', 'Missing Rate', 'Most Popular'])
#     for i, col in enumerate(cls.cat_cols_analysis):
#         missing_rate = pdf[col].isnull().mean() * 100
#         most_popular = pdf[col].value_counts().nlargest(5).index.tolist() if not pdf[col].mode().empty else np.nan
#
#         cat_analysis_pdf.loc[i, 'Column Name'] = col
#         cat_analysis_pdf.loc[i, 'Missing Rate'] = missing_rate
#         cat_analysis_pdf.loc[i, 'Most Popular'] = most_popular
#
#     print("Categorical Analysis")
#     print(cat_analysis_pdf)
#
#     unique_customers_per_month = pdf.groupby('YearMonth')['customer_code'].nunique()
#     unique_customers_per_month = unique_customers_per_month.reset_index()
#     unique_customers_per_month.columns = ['Month', 'Unique Customers']
#
#     print(unique_customers_per_month)
#
#     monthly_customers = pdf.groupby('YearMonth')['customer_code'].unique()
#     num_clients = pd.DataFrame(columns=['YearMonth', 'new_customers', 'lost_customers'])
#     for i in range(1, len(monthly_customers)):
#         prev_month_customers = set(monthly_customers[i - 1])
#         curr_month_customers = set(monthly_customers[i])
#
#         new_customers = len(curr_month_customers - prev_month_customers)
#         lost_customers = len(prev_month_customers - curr_month_customers)
#
#         num_clients.loc[i] = [monthly_customers.index[i], new_customers, lost_customers]
#
#     print('New and Lost Clients per Month')
#     print(num_clients)
#
#     monthly_assets = pdf.groupby('YearMonth')[cls.product_cols].sum()
#
#     print('Assets per Month')
#     print(monthly_assets)
#
#
# @classmethod
# def count_new_lost(cls, curr_month, prev_month):
#     new_customers = curr_month - prev_month
#     lost_customers = prev_month - curr_month if prev_month > curr_month else 0
#     return new_customers, lost_customers
#
#
# @classmethod
# def final_form(cls, pdf):
#     label_cols = [col for col in pdf.columns if '_label' in col]
#
#     pdf['next_month_products'] = pdf[label_cols].apply(lambda row: dict(row), axis=1)
#
#     pdf = pdf.drop(columns=label_cols)
#     pdf = pdf.drop(columns='grass_date')
#     pdf = pdf.drop(columns='YearMonth')
#
#     print(pdf)
#     return pdf
#
#
# @classmethod
# def statistics(cls, train_pdf, val_pdf, test_pdf):
#     stats = {}
#
#     stats['total_rows_train'] = train_pdf.shape[0]
#     stats['total_rows_val'] = val_pdf.shape[0]
#     stats['total_rows_test'] = test_pdf.shape[0]
#
#     stats['unique_customers_train'] = train_pdf['customer_code'].nunique()
#     stats['unique_customers_val'] = val_pdf['customer_code'].nunique()
#     stats['unique_customers_test'] = test_pdf['customer_code'].nunique()
#
#     stats['duplicated_rows_train'] = train_pdf.loc[:, train_pdf.columns != 'next_month_products'].duplicated().sum()
#     stats['duplicated_rows_val'] = val_pdf.loc[:, val_pdf.columns != 'next_month_products'].duplicated().sum()
#     stats['duplicated_rows_test'] = test_pdf.loc[:, test_pdf.columns != 'next_month_products'].duplicated().sum()
#
#     stats['duplicated_customers_train'] = train_pdf.duplicated(subset=['customer_code']).sum()
#     stats['duplicated_customers_val'] = val_pdf.duplicated(subset=['customer_code']).sum()
#     stats['duplicated_customers_test'] = test_pdf.duplicated(subset=['customer_code']).sum()
#
#     train_val_customers = set(train_pdf['customer_code']).union(val_pdf['customer_code'])
#
#     common_customers_train_val_test = train_val_customers.intersection(test_pdf['customer_code'])
#
#     stats['common_customers_train_val_test'] = len(common_customers_train_val_test)
#
#     print(stats)
#     return stats
#
#
# @classmethod
# def label_statistic(cls, pdf):
#     inner = pdf.copy()
#
#     for product, label in zip(cls.product_cols, cls.label_cols):
#         inner[product + '_change'] = inner[product] != inner[label]
#
#     product_change_counts = inner[[product + '_change' for product in cls.product_cols]].sum(axis=1)
#     num_customers_changed = (product_change_counts > 0).sum()
#
#     print(num_customers_changed)
#     return num_customers_changed
#
#
# @classmethod
# def final_form_analysis(cls, pdf):
#     engineered_features = [col for col in pdf.columns if col not in cls.original_columns]
#
#     num_analysis_pdf = pd.DataFrame(index=engineered_features,
#                                     columns=['Missing Rate', 'Zero Rate', 'Mean', 'Variance', '20%', '40%', '60%',
#                                              '80%', 'Min Value', 'Max Value'])
#
#     for i, col in enumerate(engineered_features):
#         missing_rate = pdf[col].isnull().mean() * 100
#         zero_rate = (pdf[col] == 0).mean() * 100
#         mean = np.mean(pdf[col])
#         variance = np.var(pdf[col])
#         quantiles = np.nanpercentile(pdf[col].dropna(), [20, 40, 60, 80])
#         min_val = np.min(pdf[col])
#         max_val = np.max(pdf[col])
#
#         num_analysis_pdf.loc[col] = [missing_rate, zero_rate, mean, variance, *quantiles, min_val, max_val]
#
#     print("Numerical Analysis")
#     print(num_analysis_pdf)

# data = Santander.load_data('train_ver2.csv')
# Santander.analyze_data(data)
# sample = Santander.sample_customers(data)
# train, validation, test = Santander.reformat(sample, '2015', '01')
# train_encoded = Santander.encoder(train)
# validation_encoded = Santander.encoder(validation)
# test_encoded = Santander.encoder(test)
# final_train = Santander.final_form(train_encoded)
# final_validation = Santander.final_form(validation_encoded)
# final_test = Santander.final_form(test_encoded)
# Santander.statistics(final_train, final_validation, final_test)
# Santander.label_statistic(train)
# Santander.label_statistic(validation)
# Santander.label_statistic(test)

# Santander.final_form_analysis(final_train)
# Santander.final_form_analysis(final_validation)
# Santander.final_form_analysis(final_test)
# Prediction.candidate_selection(final_train, 1074483)


#MODEL TESTS
# train_similar = CustomerSimilarity.similar(final_train, 92944)
# filtered = CustomerSimilarity.filter(final_train, 92944, train_similar)
# CustomerSimilarity.recommend_products(final_train, 92944)
# results = CustomerSimilarity.evaluate_recommendations(final_train, 92944, 5)
# newprod = CustomerSimilarity.customers_with_new_products(final_train)
# yaha = CustomerSimilarity.rank_recommendations(final_train, 462472,  5)
# [765322, 188637, 1263322, 534367, 867286, 1041868, 1291810, 1050326, 182012, 1105218, 49480, 540707, 461334, 1523408, 1514375, 65148, 605117, 397737, 149245, 1252575, 394814, 518977, 1234723, 1500455, 63351, 65404, 1403858, 159396, 538368, 474020, 1020806, 145693, 196730, 377771, 717147]
# [395051, 1167485, 1515755, 1406130, 982893, 1277974, 1267893]
# X, y = CustomerSimilarity.reformat(final_train, 1153296)
# model = CustomerSimilarity.train_ranking_model(X, y)
# X_val, y_val = CustomerSimilarity.reformat(final_validation, 1153296)
# predictions = model.predict(X_val)
# X_train, y_train = CustomerSimilarity.reformat_all(final_train)
# X_val, y_val = CustomerSimilarity.reformat_all(final_val)
# X_test, y_test = CustomerSimilarity.reformat_all(final_test)
# model = CustomerSimilarity.train_ranking_model(X_train, y_train)
# #test on training data
# newprod_train = CustomerSimilarity.customers_with_new_products(final_train)
# [765322, 188637, 1263322, 534367, 867286, 1041868, 1291810, 1050326, 182012, 1105218, 49480, 540707, 461334, 1523408, 1514375, 65148, 605117, 397737, 149245, 1252575, 394814, 518977, 1234723, 1500455, 63351, 65404, 1403858, 159396, 538368, 474020, 1020806, 145693, 196730, 377771, 717147]
# CustomerSimilarity.evaluate_predictions(model, X_train, y_train, 1010096, 5)
# #test on validation data
# newprod_val = CustomerSimilarity.customers_with_new_products(final_val)
# [215246, 1248920, 627386, 428664, 124110, 1063956, 674634, 1225267, 931234, 765186, 856506, 1012924, 1063674, 836265, 1331618, 498424, 93158, 1384930, 591218, 1377561, 1526646, 1115177, 102662, 1419258, 1292886, 722803, 97465, 391582, 935604, 1212934, 77823, 1501210, 943801, 657650, 550256, 155969, 634788, 524768, 1035855, 1528145, 1370630, 745195, 267092, 150406, 826805, 542916, 1221325, 891875, 690471, 207852, 430967, 689813, 1076919, 473849, 1358492, 1455376, 1022491, 892294, 362129, 1517795, 1226905, 871894, 1520901, 1093674, 1456343, 473329, 825239, 461334]
# CustomerSimilarity.evaluate_predictions(model, X_val, y_val, 1010096, 5)
# #test on test data
# newprod_test = CustomerSimilarity.customers_with_new_products(final_test)
# [1318507, 1316187, 1330137, 1330958, 1301743, 1308990, 1357262, 1369234, 1364665, 1341090, 1334232, 1229848, 1233371, 1241174, 1245543, 1235922, 1238396, 1221071, 1279366, 1277698, 1282785, 1254183, 1248361, 1253277, 1268294, 1266781, 1268133, 1269915, 1266341, 1480352, 1463171, 1466384, 1459719, 1521364, 1521376, 1519831, 1519769, 1520837, 1520867, 1524888, 1515925, 1514636, 1518711, 1518890, 1532478, 1532130, 1532176, 1533100, 1532848, 1532995, 1532975, 1531161, 1531825, 1531533, 1531560, 1535076, 1535137, 1534928, 1534979, 1534975, 1535667, 1535701, 1533734, 1534472, 1534177, 1534335, 1534286, 1527797, 1527739, 1525676, 1526404, 1526618, 1526538, 1528604, 1528784, 1501460, 1503297, 1503454, 1498864, 1511812, 1510433, 1403845, 1404788, 1398476, 1396733, 1399587, 1399840, 1417178, 1410912, 1378495, 1382559, 1376525, 1394527, 1394512, 1384930, 1385193, 1386020, 1383396, 1383736, 1384731, 1386768, 1445955, 1447622, 1442410, 1456477, 1426753, 1438491, 452257, 453606, 433365, 444777, 489206, 492707, 507550, 486190, 386048, 358598, 357816, 366251, 367061, 365729, 418662, 420956, 428113, 430425, 400440, 399404, 634593, 672595, 686138, 654346, 669451, 537364, 538198, 548782, 548457, 547557, 551441, 509913, 525030, 527650, 520164, 520136, 519924, 519077, 592432, 585019, 600222, 601752, 555237, 553903, 556888, 578757, 124757, 119612, 131087, 105404, 99023, 111968, 164395, 162884, 177807, 176833, 143355, 145134, 145052, 154357, 152554, 155339, 43201, 46142, 48356, 52123, 51046, 24631, 30599, 31490, 86772, 79779, 94921, 95627, 94555, 92396, 66527, 78605, 69226, 73683, 296667, 300617, 273535, 274575, 342225, 327351, 321144, 209195, 207839, 203954, 211760, 212778, 211084, 187963, 246703, 247512, 256987, 230197, 223044, 223295, 238557, 239387, 241067, 1065105, 1074291, 1056624, 1053657, 1053617, 1093354, 1085408, 1018522, 1019901, 1015134, 1003111, 1043571, 1032918, 1027828, 1039357, 1164590, 1187054, 1185007, 1185363, 1184197, 1191432, 1115127, 1126689, 1121894, 1109529, 1108564, 1151244, 1147651, 1157430, 1136810, 1136662, 1143379, 1141170, 824091, 832344, 834453, 805084, 799781, 797537, 875072, 868778, 885974, 852860, 844521, 725073, 741297, 732009, 731690, 705148, 713057, 718822, 770173, 770799, 770582, 789786, 753081, 956839, 987683, 993882, 991729, 993124, 975785, 982931, 940894, 935604, 928256, 924176, 931474]
# CustomerSimilarity.evaluate_predictions(model, X_test, y_test, 1010096, 5)
#
# CustomerSimilarity.average_recall(model, X_train, y_train, newprod_train)
# CustomerSimilarity.average_recall(model, X_val, y_val, newprod_val)
# CustomerSimilarity.average_recall(model, X_test, y_test, newprod_test)



#From model.py
# @classmethod
# def evaluate_recommendations(cls, pdf, customer_code, top_n=10):
#     recommended_products = cls.recommend_products(pdf, customer_code)
#     recommended_products_list = [item[1] for item in recommended_products.index.tolist()]
#
#     actual_products = pdf[pdf['customer_code'] == customer_code] \
#         [[col for col in pdf.columns if col.endswith("_label")]]
#     actual_products = actual_products.loc[:, (actual_products != 0).any(axis=0)].columns
#     actual_products = [col.replace("_label", "") for col in actual_products]
#
#     current_products = pdf[pdf['customer_code'] == customer_code] \
#         [[col for col in pdf.columns if not col.endswith('_label') and col != 'customer_code']]
#     current_products = current_products.loc[:, (current_products != 0).any(axis=0)].columns
#
#     new_products = list(set(actual_products) - set(current_products))
#
#     correct_predictions = set(recommended_products_list) & set(new_products)
#
#     precision = len(correct_predictions) / len(recommended_products_list) if recommended_products_list else 0
#     recall = len(correct_predictions) / len(new_products) if new_products else 0
#
#     print("Recommended products:", recommended_products_list)
#     print("Next month's new products:", new_products)
#     print("Precision =", precision)
#     print("Recall =", recall)

# @classmethod
# def get_top_similar_customers(cls, pdf, customer_code, top_n=10):
#     usecols = [col for col in pdf.columns if not col.endswith('_label') and col != 'customer_code']
#     similar_users = cls.similar(pdf, customer_code, usecols)
#     return similar_users.head(top_n)
#
# @classmethod
# def calculate_map_at_7(cls, model, X, y, customers):
#    sum_precision = 0
#    U = len(customers)
#
#    for customer_code in customers:
#        predictions = cls.make_predictions(model, X, customer_code)
#        top_predictions = predictions.nlargest(7, 'prediction')['product'].tolist()
#
#        actual_products_next_month = X[(X['customer_code'] == customer_code) & (y == 1)]['product'].tolist()
#        current_products = X[(X['customer_code'] == customer_code) & (X['current_ownership'] == 1)][
#            'product'].tolist()
#        new_products = list(set(actual_products_next_month) - set(current_products))
#
#        print("Customer Code:", customer_code)
#        print("Predicted Products:", top_predictions)
#        print("Actual New Products Next Month:", new_products)
#
#        m = len(new_products)
#        n = len(top_predictions)
#        precision_at_k_val = cls.precision_at_k(new_products, top_predictions, k=min(m, 7))
#
#        print("Precision at k for customer:", precision_at_k_val)
#
#        sum_precision += (1 / min(m, 7)) * precision_at_k_val
#
#    return sum_precision / U
#
#
# @classmethod
# def evaluate_predictions(cls, model, X, y, customer_code, top_n=5):
#     predictions = cls.make_predictions(model, X, customer_code)
#
#     top_predictions = predictions.nlargest(top_n, 'prediction')['product'].tolist()
#
#     actual_products_next_month = X[(X['customer_code'] == customer_code) & (y == 1)]['product'].tolist()
#
#     current_products = X[(X['customer_code'] == customer_code) & (X['current_ownership'] == 1)]['product'].tolist()
#
#     new_products = list(set(actual_products_next_month) - set(current_products))
#
#     correct_predictions = set(top_predictions) & set(new_products)
#     precision = len(correct_predictions) / len(top_predictions) if top_predictions else 0
#     recall = len(correct_predictions) / len(new_products) if new_products else 0
#
#     print("Top predicted products:", top_predictions)
#     print("Actual new products next month:", new_products)
#     print("Precision =", precision)
#     print("Recall =", recall)
#
#
# @classmethod
# def precision_at_k(cls, actual, predicted, k=7):
#     if len(predicted) > k:
#         predicted = predicted[:k]
#
#     num_hits = 0.0
#
#     for p in predicted:
#         if p in actual:
#             num_hits += 1.0
#
#     precision = num_hits / k if k > 0 else 0.0
#     print("Precision at k:", precision)
#     return precision
#
#
#
# DATA ANALYSIS HELPER:
#
#
# @classmethod
# def creation_group(cls, pdf, col):
#     pdf[col] = pd.to_datetime(pdf[col])
#     creation_year = pdf[col].dt.year
#     groups = [0, 2010, 2013, np.inf]
#     groups_labels = ['Very Old Client', 'Old Client', 'New Client']
#     pdf['account_age'] = pd.cut(creation_year, bins=groups, labels=groups_labels)