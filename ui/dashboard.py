import streamlit as st
import ast
import sys
import pandas as pd
import matplotlib.pyplot as plt
import json
st.set_option('deprecation.showPyplotGlobalUse', False)
sys.path.insert(0, '../lp-ds-recommender-system/')
from dao.santander.santander import Santander
from ui.session import get
from models.model import CustomerSimilarity



data_path = 'train_ver2.csv'

session_state = get(data=None, combined_results=None, data2=None)

if session_state.data is None:
    session_state.data = Santander.load_data_dashboard(data_path)

data = session_state.data

if session_state.data2 is None:
    session_state.data2 = Santander.load_data(data_path)

data2 = session_state.data2

if session_state.combined_results is None:
    session_state.combined_results = pd.read_csv('./pipeline/combined_results.csv')

combined_results = session_state.combined_results

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", ['Welcome', 'Customer Exploration', 'Model Performance',
                                       'Product Recommendations & Similar Customers'])


def load_data():
    if session_state.data is None:
        session_state.data = Santander.load_data_dashboard(data_path)
    return session_state.data


def welcome():
    st.title("Welcome to Julius Baer's Recommendation System")
    st.write("By Arian Madadi")
    st.write("Please use the sidebar to navigate between pages.")


def customer_exploration():
    st.title('Customer Exploration')
    data = load_data()
    st.sidebar.header('Customer Selection')
    unique_customers = data['customer_code'].unique()
    selected_customer = st.sidebar.selectbox('Choose a customer code', unique_customers)
    customer_data = data[data['customer_code'] == selected_customer]
    st.write('Data for Selected Customer:', selected_customer)
    st.write(customer_data)


def model_performance():
    st.title('Model Performance')

    with open('./pipeline/logs.json',
              'r') as file:
        logs = json.load(file)

    logs_df = pd.DataFrame(list(logs.items()), columns=["Processing Window", "Recall"])

    logs_df["Recall"] = (logs_df["Recall"] * 100).round(2)

    logs_df["Recall"] = logs_df["Recall"].astype(str) + '%'

    st.table(logs_df)

    st.header('Performance Over Time')
    recall_values = [float(val.strip('%')) for val in logs_df["Recall"]]
    window_labels = list(logs.keys())
    plt.plot(recall_values, label='Recall')
    plt.axhline(y=30, color='r', linestyle='--', label='Threshold')
    plt.legend()
    plt.xlabel('Time')
    plt.ylim(0, 100)
    plt.ylabel('Recall (%)')
    plt.xticks(ticks=range(len(logs)), labels=window_labels, rotation=45)
    st.pyplot()


def product_recommendations():
    selected_customer_from_combined = st.sidebar.selectbox('Choose a customer code from recommendations', combined_results['customer_code'].unique())

    st.header('Top Product Recommendations for Selected Customer:')
    recommended_products = combined_results.loc[combined_results['customer_code'] == selected_customer_from_combined, 'top_predicted_products'].iloc[0]

    if isinstance(recommended_products, str):
        recommended_products = ast.literal_eval(recommended_products)

    formatted_products = "\n".join([f"- {product}" for product in recommended_products])
    st.markdown(formatted_products)

    combined_customer_codes = combined_results['customer_code'].unique()
    filtered_data = data2[data2['customer_code'].isin(combined_customer_codes)]
    final_train, final_val, final_test = Santander.prepare_data(filtered_data, '2015', '01')
    final = pd.concat([final_train, final_val], ignore_index=True)

    st.write('Data for Selected Customer:', selected_customer_from_combined)

    top_similar_customers = CustomerSimilarity.similar(final, selected_customer_from_combined).head(10)
    top_similar_customers.reset_index(drop=True, inplace=True)
    top_similar_customers.index = range(1, 11)

    st.header('Top 10 Most Similar Customers to the Selected Customer:')
    st.table(top_similar_customers)


if selection == 'Welcome':
    welcome()
elif selection == 'Customer Exploration':
    customer_exploration()
elif selection == 'Model Performance':
    model_performance()
elif selection == 'Product Recommendations & Similar Customers':
    product_recommendations()
