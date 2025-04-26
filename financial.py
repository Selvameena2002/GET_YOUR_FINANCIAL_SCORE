import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Home", "Score", "Bar chart", "Pie chart", "Scatter plot", "Line plot", "Box Plot"])

with tab1:
    st.header('FINANCIAL HEALTH CHECK APP')
    st.subheader('Welcome To Financial Health Check App')
    st.markdown("A financial health check helps assess an organization's profitability and sustainability by analyzing key financial metrics. The primary focus is on determining whether the company is making a profit or loss.")

with tab2:
    dataset = st.sidebar.file_uploader("Upload your CSV file here", type='csv')

    if dataset is not None:
        data = pd.read_csv(dataset)

        def calculate_profit_loss(row):
            profit_value = row['Selling_Price'] - row['Actual_Price']
            loss_value = row['Actual_Price'] - row['Selling_Price']
            if profit_value > 0:
                return profit_value, 0
            else:
                return 0, loss_value

        data[['profit_value', 'loss_value']] = data.apply(calculate_profit_loss, axis=1, result_type='expand')

        st.dataframe(data)

        data = data.fillna(0).apply(pd.to_numeric, errors='ignore').drop_duplicates()

        categorical_columns = data.select_dtypes(include=['object']).columns
        label_encoders = {}
        product_name_mapping = {}

        for col in categorical_columns:
            label_encoders[col] = LabelEncoder()
            data[col] = label_encoders[col].fit_transform(data[col])
            if col == 'Product_Name':
                product_name_mapping = dict(zip(data[col], label_encoders[col].classes_))

        st.session_state.product_name_mapping = product_name_mapping

        X = data.iloc[:, :4]  
        y = data.iloc[:, 4:]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

        model = RandomForestRegressor(n_estimators=10)
        model.fit(X_train, y_train)

        st.success('Model training is done')

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        st.write(f"R^2 Score: {r2}")
        st.write(f"Mean Absolute Error: {mae}")

        st.session_state.data = data  

    else:
        st.warning('Please upload your CSV file.')

with tab3: 
    if 'data' in st.session_state:
        data = st.session_state.data
        st.subheader('Bar Chart')
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Product_Name', y='profit_value', data=data, color='green', label='Profit')
        sns.barplot(x='Product_Name', y='loss_value', data=data, color='red', label='Loss')
        plt.legend()
        st.pyplot()

        if 'product_name_mapping' in st.session_state:
            st.subheader("Products with Profit")
            profitable_products = data[data['profit_value'] > 0]['Product_Name'].unique()
            for encoded in profitable_products:
                if encoded in st.session_state.product_name_mapping:
                    name = st.session_state.product_name_mapping[encoded]
                    st.markdown(f"**Product {encoded}**: {name}")
    else:
        st.warning('Please upload a CSV file in the "Score" tab first.')

with tab4: 
    if 'data' in st.session_state:
        data = st.session_state.data
        st.subheader('Pie Chart')
        plt.figure(figsize=(7, 7))
        profit_loss = data[['profit_value', 'loss_value']].sum()
        plt.pie(profit_loss, labels=profit_loss.index, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        st.pyplot()

        if 'product_name_mapping' in st.session_state:
            st.subheader("Products with Profit")
            profitable_products = data[data['profit_value'] > 0]['Product_Name'].unique()
            for encoded in profitable_products:
                if encoded in st.session_state.product_name_mapping:
                    name = st.session_state.product_name_mapping[encoded]
                    st.markdown(f"**Product {encoded}**: {name}")
    else:
        st.warning('Please upload a CSV file in the "Score" tab first.')

with tab5: 
    if 'data' in st.session_state:
        data = st.session_state.data
        st.subheader('Scatter Plot')
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x='Product_Name', y='profit_value', data=data, color='green', label='Profit')
        sns.scatterplot(x='Product_Name', y='loss_value', data=data, color='red', label='Loss')
        plt.legend()
        st.pyplot()

        if 'product_name_mapping' in st.session_state:
            st.subheader("Products with Profit")
            profitable_products = data[data['profit_value'] > 0]['Product_Name'].unique()
            for encoded in profitable_products:
                if encoded in st.session_state.product_name_mapping:
                    name = st.session_state.product_name_mapping[encoded]
                    st.markdown(f"**Product {encoded}**: {name}")
    else:
        st.warning('Please upload a CSV file in the "Score" tab first.')

with tab6:
    if 'data' in st.session_state:
        data = st.session_state.data
        st.subheader('Line Chart')
        plt.figure(figsize=(8, 5))
        sns.lineplot(x='Product_Name', y='profit_value', data=data, marker='o', label='Profit')
        sns.lineplot(x='Product_Name', y='loss_value', data=data, marker='o', label='Loss')
        plt.legend()
        st.pyplot()

        if 'product_name_mapping' in st.session_state:
            st.subheader("Products with Profit")
            profitable_products = data[data['profit_value'] > 0]['Product_Name'].unique()
            for encoded in profitable_products:
                if encoded in st.session_state.product_name_mapping:
                    name = st.session_state.product_name_mapping[encoded]
                    st.markdown(f"**Product {encoded}**: {name}")
    else:
        st.warning('Please upload a CSV file in the "Score" tab first.')

with tab7:
    if 'data' in st.session_state:
        data = st.session_state.data
        st.subheader('Box Plot')
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='Product_Name', y='profit_value', data=data, color='green')
        sns.boxplot(x='Product_Name', y='loss_value', data=data, color='red')
        st.pyplot()

        if 'product_name_mapping' in st.session_state:
            st.subheader("Products with Profit")
            profitable_products = data[data['profit_value'] > 0]['Product_Name'].unique()
            for encoded in profitable_products:
                if encoded in st.session_state.product_name_mapping:
                    name = st.session_state.product_name_mapping[encoded]
                    st.markdown(f"**Product {encoded}**: {name}")
    else:
        st.warning('Please upload a CSV file in the "Score" tab first.')
