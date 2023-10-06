import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from dao.santander.santander_definitions import product_cols



class CustomerSimilarity:


   product_dict = {i: product for i, product in enumerate(product_cols)}


   @classmethod
   def similar(cls, pdf, customer_code, users=1000):
      usecols = [col for col in pdf.columns if not col.endswith('_label') and col != 'customer_code']

      user_product_matrix = pdf[usecols].values
      target_user_vector = pdf[pdf['customer_code'] == customer_code][usecols].values

      similarity_scores = cosine_similarity(user_product_matrix, target_user_vector)
      similarity_scores = similarity_scores.flatten()

      similar_user_indices = similarity_scores.argsort()[-users - 2:-1][::-1]

      similar_users = pdf.iloc[similar_user_indices]['customer_code']
      similar_users_scores = similarity_scores[similar_user_indices]

      # for user, score in zip(similar_users, similar_users_scores):
      #    print(f"Similarity score between user {customer_code} and user {user}: {score}")

      return similar_users


   @classmethod
   def filter(cls, pdf, original, candidates):
      original_products = pdf[pdf['customer_code'] == original][product_cols].values

      candidates_list = candidates.tolist()
      for candidate in candidates_list:
         candidate_products = pdf[pdf['customer_code'] == candidate][product_cols].values
         if np.array_equal(original_products, candidate_products):
            candidates_list.remove(candidate)
         else:
            diff_products = np.setdiff1d(candidate_products, original_products)
            if len(diff_products) == 0:
               candidates_list.remove(candidate)

      filtered_candidates = pd.Series(candidates_list)
      return filtered_candidates


   @classmethod
   def recommend_products(cls, pdf, customer_code):
      similar_users = cls.similar(pdf, customer_code)
      filtered_users = cls.filter(pdf, customer_code, similar_users)

      product_vectors = pdf[pdf['customer_code'].isin(filtered_users)][product_cols]
      mean_product_vector = product_vectors.mean()

      user_products = pdf[pdf['customer_code'] == customer_code][product_cols]
      recommended_products = mean_product_vector - user_products

      recommended_products_series = recommended_products.stack()
      return recommended_products_series


   @classmethod
   def customers_with_new_products(cls, pdf):
      customer_codes = []

      for customer_code in pdf['customer_code'].unique():
         actual_products = pdf[pdf['customer_code'] == customer_code][[col for col in pdf.columns if col.endswith("_label")]]
         actual_products = actual_products.loc[:, (actual_products != 0).any(axis=0)].columns
         actual_products = [col.replace("_label", "") for col in actual_products]

         current_products = pdf[pdf['customer_code'] == customer_code][[col for col in pdf.columns if not col.endswith('_label') and col != 'customer_code']]
         current_products = current_products.loc[:, (current_products != 0).any(axis=0)].columns

         new_products = set(actual_products) - set(current_products)
         if len(new_products) > 0:
            customer_codes.append(customer_code)

      return customer_codes


   @classmethod
   def sanitize_column_names(cls, pdf):
       pdf.columns = pdf.columns.str.replace('[<,>,:,",|,?,*, ,\n,\t]', '', regex=True)
       return pdf


   @classmethod
   def reformat(cls, pdf, customer_code, prediction_scores, query_id):
       customer_data = pdf[pdf['customer_code'] == customer_code]

       customer_features = customer_data.drop(
           columns=[col for col in pdf.columns if col.endswith('_label') or col in product_cols]).iloc[0]

       matrix = []
       target_vector = []

       for product in product_cols:
           current_ownership = customer_data[product].iloc[0]
           future_ownership = customer_data[product + '_label'].iloc[0]

           row = customer_features.to_dict()
           row['product'] = product
           row['current_ownership'] = current_ownership
           row['prediction_score'] = prediction_scores.get(product, 0)
           row['query_id'] = query_id
           matrix.append(row)

           target_vector.append((product, future_ownership))

       matrix = pd.DataFrame(matrix)
       target_vector = pd.Series(dict(target_vector), name='future_ownership')

       return matrix, target_vector


   @classmethod
   def reformat_all(cls, pdf):
       binary_matrix_list = []
       target_vector_list = []

       query_id = 0

       for customer_code in pdf['customer_code'].unique():
           try:
               recommended_products = cls.recommend_products(pdf, customer_code)

               prediction_scores = {product: score for (_, product), score in recommended_products.items()}

               binary_matrix, target_vector = cls.reformat(pdf, customer_code, prediction_scores, query_id)

               binary_matrix_list.append(binary_matrix)
               target_vector_list.append(target_vector)

               query_id += 1
           except Exception as e:
               print(f"Skipping customer {customer_code} due to error: {e}")

       if not binary_matrix_list:
           return None, None

       binary_matrix_all = pd.concat(binary_matrix_list, ignore_index=True)
       target_vector_all = pd.concat(target_vector_list, ignore_index=True)

       binary_matrix_all = CustomerSimilarity.sanitize_column_names(binary_matrix_all)

       return binary_matrix_all, target_vector_all


   @classmethod
   def train_ranking_model(cls, X_train, y_train):
       X_use = X_train.copy()
       X_use['product'] = X_use['product'].map({v: k for k, v in cls.product_dict.items()})

       groups = X_use['customer_code'].value_counts().sort_index().values

       categorical_feature = ['product']

       model = LGBMRanker()
       model.fit(X_use.drop(columns='customer_code'), y_train, group=groups, categorical_feature=categorical_feature)

       return model

    
   @classmethod
   def train_logistic_regression_model(cls, X_train, y_train):
       X_use = X_train.copy()
       X_use['product'] = X_use['product'].map({v: k for k, v in cls.product_dict.items()})

       model = LogisticRegression()
       model.fit(X_use.drop(columns='customer_code'), y_train)

       return model


   @classmethod
   def randomizer(cls, top_n_predictions, customer_data, n):
       top_n_predictions = top_n_predictions.copy()
       if np.random.rand() < 0.2:
           not_owned = customer_data[customer_data['current_ownership'] == 0]['product'].tolist()

           if not_owned and n <= len(top_n_predictions):
               random_product = np.random.choice(not_owned)

               top_n_predictions.loc[top_n_predictions.index[n - 1], 'product'] = random_product

       return top_n_predictions


   @classmethod
   def make_predictions(cls, model, X, customer_code, top_n=5):
       customer_data = X[X['customer_code'] == customer_code].copy()
       customer_data['product'] = customer_data['product'].map({v: k for k, v in cls.product_dict.items()})

       customer_data_without_code = customer_data.drop(columns=['customer_code'])
       predictions = model.predict(customer_data_without_code)

       customer_data['prediction'] = predictions
       customer_data['product'] = customer_data['product'].map(cls.product_dict)

       recommendable_products = customer_data[customer_data['current_ownership'] == 0]

       sorted_data = recommendable_products.sort_values('prediction', ascending=False)
       top_n_predictions = sorted_data.head(top_n)
       randomized_predictions = cls.randomizer(top_n_predictions, customer_data, top_n)

       # return sorted_data[['product', 'prediction']]
       return randomized_predictions[['product', 'prediction']]


   @classmethod
   def evaluate_all_customers(cls, model, X, y):
       customer_codes = []
       top_predicted_products_list = []
       actual_new_products_list = []
       precisions = []
       recalls = []

       for customer_code in X['customer_code'].unique():
           try:
               predictions = cls.make_predictions(model, X, customer_code)
               top_predictions = predictions['product'].tolist()
               # top_predictions = predictions.nlargest(top_n, 'prediction')['product'].tolist()

               actual_products_next_month = X[(X['customer_code'] == customer_code) & (y == 1)]['product'].tolist()
               current_products = X[(X['customer_code'] == customer_code) & (X['current_ownership'] == 1)][
                   'product'].tolist()
               new_products = list(set(actual_products_next_month) - set(current_products))

               correct_predictions = set(top_predictions) & set(new_products)
               precision = len(correct_predictions) / len(top_predictions) if top_predictions else 0
               recall = len(correct_predictions) / len(new_products) if new_products else 0

               customer_codes.append(customer_code)
               top_predicted_products_list.append(top_predictions)
               actual_new_products_list.append(new_products)
               precisions.append(precision)
               recalls.append(recall)

           except Exception as e:
               print(f"Error evaluating customer {customer_code}: {e}")

       results_df = pd.DataFrame({
           'customer_code': customer_codes,
           'top_predicted_products': top_predicted_products_list,
           'actual_new_products_next_month': actual_new_products_list,
           'precision': precisions,
           'recall': recalls
       })

       return results_df


   @classmethod
   def average_recall(cls, model, X, y, customers, top_n=5):
       recalls = []
       for customer_code in customers:
           predictions = cls.make_predictions(model, X, customer_code)

           top_predictions = predictions.nlargest(top_n, 'prediction')['product'].tolist()

           actual_products_next_month = X[(X['customer_code'] == customer_code) & (y == 1)]['product'].tolist()

           current_products = X[(X['customer_code'] == customer_code) & (X['current_ownership'] == 1)][
               'product'].tolist()

           new_products = list(set(actual_products_next_month) - set(current_products))

           correct_predictions = set(top_predictions) & set(new_products)
           recall = len(correct_predictions) / len(new_products) if new_products else 0

           recalls.append(recall)

       return np.mean(recalls)

