import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns

st.set_page_config(
    page_title="Hotel Booking Analysis",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data():
    data = pd.read_csv("Dataset/hotel_bookings.csv")
    return data

data = load_data()
st.markdown("""
    <style>

    /* Title Styling */
    .title {
        font-size: 30px;
        font-weight: bold;
        color: #87bdd8 !important;
    }

    /* Subtitle Styling */
    .subtitle {
        font-size: 20px;
        font-weight: bold;
        color: #b7d7e8 !important;
    }

    /* Section Text Styling */
    .section {
        font-size: 16px;
        font-weight: normal;
        color: #87bdd8 !important;
    }

    /* Sidebar Styling */
    .css-1d391kg {  /* Sidebar Background */
        background-color: #b7d7e8 !important;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #cfe0e8 !important;
        color: #87bdd8 !important;
        font-weight: bold;
        border-radius: 5px;
        border: none;
    }

    .stButton>button:hover {
        background-color: #87bdd8 !important;
        color: #daebe8 !important;
    }

    /* Tabs Styling */
    .stTabs .stTab {
        background-color: #cfe0e8 !important;
        color: #87bdd8 !important;
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }

    .stTabs .stTab:hover {
        background-color: #87bdd8 !important;
        color: #daebe8 !important;
    }

    .stTabs .stTab.stTabActive {
        background-color: #87bdd8 !important;
        color: #daebe8 !important;
        font-weight: bold !important;
    }

    /* Table Header and Data */
    .stTable th {
        background-color: #cfe0e8 !important;
        color: #87bdd8 !important;
        padding: 10px;
    }

    .stTable td {
        background-color: #daebe8 !important;
        color: #87bdd8 !important;
        padding: 10px;
    }

    .stTable tr:hover {
        background-color: #b7d7e8 !important;
        color: #daebe8 !important;
    }

    /* Plotly Chart Styling */
    .plotly-graph-div {
        border-radius: 10px !important;
        background-color: #daebe8 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def preprocess_data(data):
    data.fillna(data.median(numeric_only=True), inplace=True)
    fc = ["hotel", "market_segment", "is_repeated_guest", 
                    "previous_cancellations", "previous_bookings_not_canceled", "customer_type"]
    tc = "is_canceled"
    cat_cols = [col for col in fc if data[col].dtype == 'object']
    encoder = ce.OrdinalEncoder(cols=cat_cols)
    data[cat_cols] = encoder.fit_transform(data[cat_cols])
    num_cols = ["is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled"]
    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])
    
    X = data[fc]
    y = data[tc].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    
    return X_train, X_test, y_train, y_test, encoder, scaler, cat_cols, num_cols

X_train, X_test, y_train, y_test, encoder, scaler, categorical_cols, numerical_cols = preprocess_data(data)

@st.cache_data
def train_model(X_train, y_train):
    model = LogisticRegression(random_state=120, max_iter=1000)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)


with st.sidebar:
    selected_section = st.radio(
        "Navigate to Section",
        ["Introduction", "EDA", "Model", "Conclusion"]
    )

if selected_section == "Introduction":
    st.title("Hotel Booking Analysis")
    st.subheader("Project Objectives")
    st.markdown("""
    - **Analyze patterns in hotel bookings** to understand factors like cancellations, trends, and customer behavior.
    - **Explore dataset insights** through visualizations and summary statistics.
    - **Build a Logistic Regression model** to predict booking cancellations.
    """)
    st.subheader("Dataset Overview")
    st.write(data.head())


elif selected_section == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    eda_tab = st.selectbox(
        "Choose an analysis",
        [
            "Overview", "Booking Cancellations", "ADR Analysis", "Monthly Trends", 
            "Guest Country Analysis", "Stay Length Analysis", "High ADR Analysis",
            "Lead Time Analysis", "Hotel Type Insights", "Special Requests",
            "Weekend vs Weekday Stays", "Market Segment Insights",
            "Deposit Type Analysis", "Customer Type Insights" , "Correlation Heatmap"
        ]
    )

    if eda_tab == "Overview":
        st.subheader("Dataset Overview")
        st.write(data.describe())

    elif eda_tab == "Booking Cancellations":
        st.subheader("Booking Cancellations")
        cancellation_rate = data['is_canceled'].value_counts(normalize=True) * 100
        st.write(f"Cancellation Rate: {round(cancellation_rate[1], 2)}%")
        fig = px.pie(
            values=cancellation_rate.values,
            names=["Not Canceled", "Canceled"],
            title="Booking Cancellation Distribution",
            color_discrete_sequence=px.colors.qualitative.G10
        )
        st.plotly_chart(fig, use_container_width=True)

    elif eda_tab == "ADR Analysis":
        st.subheader("Average Daily Rate (ADR) Analysis")
        fig = px.box(data, x='hotel', y='adr', color='hotel', title="ADR Distribution by Hotel")
        st.plotly_chart(fig, use_container_width=True)

    elif eda_tab == "Monthly Trends":
        st.subheader("Monthly Booking Trends")
        data['arrival_date'] = pd.to_datetime(data['arrival_date_year'].astype(str) + "-" + data['arrival_date_month'])
        monthly_trends = data.groupby('arrival_date')['hotel'].count().reset_index()
        fig = px.line(monthly_trends, x='arrival_date', y='hotel', title="Monthly Booking Trends")
        st.plotly_chart(fig, use_container_width=True)

    elif eda_tab == "Guest Country Analysis":
        st.subheader("Guest Country Analysis")
        country_data = data['country'].value_counts().reset_index().head(10)
        country_data.columns = ['Country', 'Bookings']
        fig = px.bar(country_data, x='Country', y='Bookings', title="Top 10 Guest Countries")
        st.plotly_chart(fig, use_container_width=True)

    elif eda_tab == "Stay Length Analysis":
        st.subheader("Stay Length Analysis")
        data['total_stay'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
        fig = px.histogram(data, x='total_stay', nbins=20, title="Total Stay Length Distribution")
        st.plotly_chart(fig, use_container_width=True)

    elif eda_tab == "High ADR Analysis":
        st.subheader("High ADR Analysis")
        high_adr = data[data['adr'] > data['adr'].quantile(0.95)]
        fig = px.scatter(
            high_adr, x='adr', y='total_of_special_requests', 
            color='hotel', title="High ADR Bookings and Special Requests"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif eda_tab == "Lead Time Analysis":
        st.subheader("Lead Time Analysis")
        fig = px.histogram(data, x='lead_time', nbins=30, title="Lead Time Distribution")
        st.plotly_chart(fig, use_container_width=True)

    elif eda_tab == "Hotel Type Insights":
        st.subheader("Hotel Type Insights")
        hotel_type_counts = data['hotel'].value_counts()
        fig = px.bar(hotel_type_counts, x=hotel_type_counts.index, y=hotel_type_counts.values, 
                     title="Hotel Type Distribution", labels={'x': "Hotel Type", 'y': "Count"})
        st.plotly_chart(fig, use_container_width=True)

    elif eda_tab == "Special Requests":
        st.subheader("Special Requests Analysis")
        fig = px.histogram(
            data, x='total_of_special_requests', nbins=5, 
            title="Special Requests Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif eda_tab == "Weekend vs Weekday Stays":
        st.subheader("Weekend vs Weekday Stays")
        fig = px.bar(
            data, x='hotel', y=['stays_in_weekend_nights', 'stays_in_week_nights'],
            title="Weekend vs Weekday Stay Comparison", barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

    elif eda_tab == "Market Segment Insights":
        st.subheader("Market Segment Insights")
        segment_counts = data['market_segment'].value_counts()
        fig = px.bar(
            segment_counts, x=segment_counts.index, y=segment_counts.values, 
            title="Market Segment Distribution", labels={'x': "Market Segment", 'y': "Count"}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif eda_tab == "Deposit Type Analysis":
        st.subheader("Deposit Type Analysis")
        deposit_counts = data['deposit_type'].value_counts()
        fig = px.pie(
            values=deposit_counts.values, names=deposit_counts.index, 
            title="Deposit Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    elif eda_tab == "Customer Type Insights":
        st.subheader("Customer Type Insights")
        customer_counts = data['customer_type'].value_counts()
        fig = px.bar(
            customer_counts, x=customer_counts.index, y=customer_counts.values, 
            title="Customer Type Distribution", labels={'x': "Customer Type", 'y': "Count"}
        )
        st.plotly_chart(fig, use_container_width=True)
    elif eda_tab == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        numerical_data = data.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numerical_data.corr()
        plt.figure(figsize=(12, 8))

        heatmap = sns.heatmap(
            correlation_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap='coolwarm', 
            cbar=True, 
            square=True, 
            linewidths=0.5,
            annot_kws={"size": 10}
        )

        plt.title("Correlation Heatmap", fontsize=16)
        
        st.pyplot(plt)

elif selected_section == "Model":
    st.title("Booking Cancellation Prediction Model")
    st.subheader("Provide Input for Prediction")

  
    hotel = st.selectbox("Hotel Type", ["City Hotel", "Resort Hotel"])
    market_segment = st.selectbox("Market Segment", ["Online TA", "Offline TA/TO", "Direct", "Corporate", "Complementary", "Groups", "Undefined"])
    is_repeated_guest = st.selectbox("Is Repeated Guest", [0, 1])
    previous_cancellations = st.number_input("Previous Cancellations", min_value=0, step=1)
    previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", min_value=0, step=1)
    customer_type = st.selectbox("Customer Type", ["Transient", "Contract", "Transient-Party", "Group"])


    user_input = pd.DataFrame({
        "hotel": [hotel],
        "market_segment": [market_segment],
        "is_repeated_guest": [is_repeated_guest],
        "previous_cancellations": [previous_cancellations],
        "previous_bookings_not_canceled": [previous_bookings_not_canceled],
        "customer_type": [customer_type]
    })


    user_input[categorical_cols] = encoder.transform(user_input[categorical_cols])
    user_input[numerical_cols] = scaler.transform(user_input[numerical_cols])

    if st.button("Predict"):
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)[0]

        if prediction[0] == 1:
            st.error(f"The booking is likely to be canceled with a probability of {np.max(prediction_proba) * 100:.2f}%.")
        else:
            st.success(f"The booking is likely to NOT be canceled with a probability of {np.max(prediction_proba) * 100:.2f}%.")

elif selected_section == "Conclusion":
    st.title("Conclusion")
    st.markdown("""
                
    - **Key Insights**:
        - High cancellation rates for certain customer types.
        - ADR differs significantly between hotel types.
        - Lead time impacts booking behavior.
    - **Model Performance**: The logistic regression model predicts cancellations effectively.
    """)


st.markdown(
    """
    <hr style='border: 1px solid #e0e0e0;'>
    <footer class='footer'>
        <p>Hotel Booking App - Created by Eman Asif</p>
    </footer>
    """,
    unsafe_allow_html=True
)