from dao.santander.santander import Santander
from models.model import CustomerSimilarity
import pandas as pd



#import data
data = Santander.load_data('train_ver2.csv')
print('imported data')

#Run this line if you want to use recsys similar to bank's original data
# data2 = Santander.benchmark(data)

#create sample
sample = Santander.sample_customers(data)
print('sampled')


#prepare and split data
final_train, final_val, final_test = Santander.prepare_data(sample, '2015', '01')
print('final splits created')


#get data ready for model
X_train, y_train = CustomerSimilarity.reformat_all(final_train)
X_val, y_val = CustomerSimilarity.reformat_all(final_val)
X_test, y_test = CustomerSimilarity.reformat_all(final_test)


#train model
model = CustomerSimilarity.train_ranking_model(X_train, y_train)


#trian on this model if you want to use recsys similar to bank's original model
# model = CustomerSimilarity.train_logistic_regression_model(X_train, y_train)


#find list of customers with new products next month
newprod_train = CustomerSimilarity.customers_with_new_products(final_train)
newprod_val = CustomerSimilarity.customers_with_new_products(final_val)
newprod_test = CustomerSimilarity.customers_with_new_products(final_test)


#test on customers from that list
recall_train = CustomerSimilarity.average_recall(model, X_train, y_train, newprod_train)
print("Recall train:")
print(recall_train)


recall_val = CustomerSimilarity.average_recall(model, X_val, y_val, newprod_val)
print("Recall validation:")
print(recall_val)


recall_test = CustomerSimilarity.average_recall(model, X_test, y_test, newprod_test)
print("Recall test:")
print(recall_test)


#creating results for display in dashboard
train_results = CustomerSimilarity.evaluate_all_customers(model, X_train, y_train)
val_results = CustomerSimilarity.evaluate_all_customers(model, X_val, y_val)
combined_results = pd.concat([train_results, val_results], ignore_index=True)


combined_results.to_csv("combined_results.csv", index=False)