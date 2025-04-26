pip install -r requirements.txt
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Set page config
st.set_page_config(page_title="Financial Health Check App", layout="wide")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Home", "Score", "Bar Chart", "Pie Chart", "Scatter Plot", "Line Plot", "Box Plot"]
)

# --- Tab 1: Home ---
with tab1:
    st.header('FINANCIAL HEALTH CHECK APP')
    st.subheader('Welcome To Financial Health Check App')
    st.markdown("""
    A financial health check helps assess an organization's profitability and sustainability 
    by analyzing key financial metrics. The primary focus is on determining 
    whether the company is making a profit or loss.
    """)

# --- Tab 2: Score ---
with tab2:
    st.sidebar.title("Upload Section")
    dataset = st.sidebar.file_uploader("Upload your CSV file here", type='csv')

    if dataset is not None:
        try:
            data = pd.read_csv(dataset)

            # Check if required columns exist
            required_columns = {'Selling_Price', 'Actual_Price', 'Product_Name'}
            if not required_columns.issubset(data.columns):
                st.error(f"CSV must contain columns: {required_columns}")
            else:
                # Calculate Profit and Loss
                def calculate_profit_loss(row):
                    profit = max(row['Selling_Price'] - row['Actual_Price'], 0)
                    loss = max(row['Actual_Price'] - row['Selling_Price'], 0)
                    return profit, loss

                data[['profit_value', 'loss_value']] = data.apply(calculate_profit_loss, axis=1, result_type='expand')

                st.dataframe(data)

                # Encode categorical features
                label_encoders = {}
                product_name_mapping = {}

                for col in data.select_dtypes(include='object').columns:
                    le = LabelEncoder()
                    data[col + '_label'] = le.fit_transform(data[col])
                    label_encoders[col] = le
                    if col == 'Product_Name':
                        product_name_mapping = dict(zip(data[col + '_label'], data[col]))

                st.session_state.product_name_mapping = product_name_mapping

                # Features and Target
                X = data[['Selling_Price', 'Actual_Price', 'Product_Name_label', 'Other_Feature1_label' if 'Other_Feature1' in data.columns else 'Product_Name_label']]
                y = data[['profit_value', 'loss_value']]

                # Train-Test Split
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

                # Model Training
                model = RandomForestRegressor(n_estimators=10, random_state=42)
                model.fit(X_train, y_train)

                st.success('Model training is complete.')

                # Evaluation
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                st.write(f"**R² Score:** {r2:.2f}")
                st.write(f"**Mean Absolute Error:** {mae:.2f}")

                st.session_state.data = data

        except Exception as e:
            st.error(f"Error loading file: {e}")

    else:
        st.warning('Please upload your CSV file.')

# --- Helper for plotting ---
def decode_product_name(data):
    if 'product_name_mapping' in st.session_state:
        data['Product_Name'] = data['Product_Name_label'].map(st.session_state.product_name_mapping)
    return data

# --- Tab 3: Bar Chart ---
with tab3:
    if 'data' in st.session_state:
        data = decode_product_name(st.session_state.data.copy())
        st.subheader('Bar Chart - Profit and Loss by Product')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Product_Name', y='profit_value', data=data, color='green', label='Profit')
        sns.barplot(x='Product_Name', y='loss_value', data=data, color='red', label='Loss')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)
    else:
        st.warning('Please upload a CSV file in the "Score" tab first.')

# --- Tab 4: Pie Chart ---
with tab4:
    if 'data' in st.session_state:
        data = st.session_state.data
        st.subheader('Pie Chart - Overall Profit vs Loss')
        fig, ax = plt.subplots(figsize=(7, 7))
        profit_loss = data[['profit_value', 'loss_value']].sum()
        ax.pie(profit_loss, labels=profit_loss.index, autopct='%1.1f%%', startangle=90, colors=['green', 'red'])
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.warning('Please upload a CSV file in the "Score" tab first.')

# --- Tab 5: Scatter Plot ---
with tab5:
    if 'data' in st.session_state:
        data = decode_product_name(st.session_state.data.copy())
        st.subheader('Scatter Plot - Profit and Loss by Product')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Product_Name', y='profit_value', data=data, color='green', label='Profit')
        sns.scatterplot(x='Product_Name', y='loss_value', data=data, color='red', label='Loss')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)
    else:
        st.warning('Please upload a CSV file in the "Score" tab first.')

# --- Tab 6: Line Chart ---
with tab6:
    if 'data' in st.session_state:
        data = decode_product_name(st.session_state.data.copy())
        st.subheader('Line Chart - Profit and Loss Over Products')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='Product_Name', y='profit_value', data=data, marker='o', label='Profit', color='green')
        sns.lineplot(x='Product_Name', y='loss_value', data=data, marker='o', label='Loss', color='red')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)
    else:
        st.warning('Please upload a CSV file in the "Score" tab first.')

# --- Tab 7: Box Plot ---
with tab7:
    if 'data' in st.session_state:
        data = decode_product_name(st.session_state.data.copy())
        st.subheader('Box Plot - Profit and Loss Distribution')
        melted = data.melt(id_vars='Product_Name', value_vars=['profit_value', 'loss_value'], var_name='Type', value_name='Value')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Product_Name', y='Value', hue='Type', data=melted)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning('Please upload a CSV file in the "Score" tab first.')
