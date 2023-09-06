import pandas as pd
import datetime
import math
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Load your data
df = pd.read_csv('Sales1.csv')

# Remove rows with a volume of 0
df = df[df['Volume'] != 0]

# Group by 'Customer Name' and aggregate the data
df1 = df.groupby('Customer Name').agg({
    'Value': 'sum',
    'Volume': 'sum',
    'Price(AED)': 'mean',
  
}).reset_index()


# Streamlit app
st.title("Customer Segmentation App")

logo_image= "https://tajgroupholding.com/cdn/shop/files/a5a99ea6-e35e-4c61-9d5c-d73392c89e40_160x160@2x.jpg?v=1670953223"
st.sidebar.image(logo_image, width=75)
# Create tabs
tabs = ["RFM Clustering", "Purchasing Behavior Clustering"]
tab_selected = st.sidebar.selectbox("Select Analysis:", tabs)

if tab_selected == "RFM Clustering":
    # Convert 'Invoice_Date' column to datetime format
    df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'])

    # Calculate reference date as the maximum Invoice_Date + 1 day
    reference_date = df['Invoice_Date'].max() + datetime.timedelta(days=1)

    # Calculate days since last purchase
    df['days_since_last_purchase'] = (reference_date - df['Invoice_Date']).dt.days

    # Calculate Recency for each customer
    customer_recency_df = df.groupby('Customer Name')['days_since_last_purchase'].min().reset_index()
    customer_recency_df.rename(columns={'days_since_last_purchase': 'recency'}, inplace=True)

    # Calculate Frequency for each customer
    customer_freq = df.groupby(['Customer Name', 'Invoice_No']).size().reset_index(name='frequency')
    customer_freq = customer_freq.groupby('Customer Name')['frequency'].count().reset_index()

    # Calculate Monetary Value for each customer
    customer_monetary_val = df.groupby("Customer Name")['Value'].sum().reset_index()

    # Merge recency, frequency, and monetary value data
    customer_history_df = customer_recency_df.merge(customer_freq, on='Customer Name')
    customer_history_df = customer_history_df.merge(customer_monetary_val, on='Customer Name')

    # Calculate logarithms for recency, frequency, and amount
    customer_history_df['recency_log'] = customer_history_df['recency'].apply(lambda x: math.log(x) if x > 0 else 0)
    customer_history_df['frequency_log'] = customer_history_df['frequency'].apply(lambda x: math.log(x) if x > 0 else 0)
    customer_history_df['amount_log'] = customer_history_df['Value'].apply(lambda x: math.log(x) if x > 0 else 0)

    # Define feature vector for RFM
    feature_vector_rfm = ['recency_log', 'frequency_log', 'amount_log']

       # Apply K-means clustering for RFM analysis
    K_best_rfm = 5  # Based on your analysis
    model_rfm = KMeans(n_clusters=K_best_rfm, random_state=101)
    model_rfm.fit(customer_history_df[feature_vector_rfm])
    
    # Create a dictionary to map original cluster labels to desired segment names
    cluster_to_segment_mapping_rfm = {
        0: "Regular Shoppers",
        1: "Occasional Shoppers",
        2: "High-Value and Frequency Shoppers",
        3: "Balanced Shoppers",
        4: "Moderate-Frequency High-Value Shoppers"
    }
    
    # Map the original cluster labels to segment names
    customer_history_df['Segment_rfm'] = model_rfm.labels_.map(cluster_to_segment_mapping_rfm)
    
    # Define consistent cluster colors based on the desired order
    cluster_colors_rfm = {
        "Regular Shoppers": 'green',
        "Occasional Shoppers": 'yellow',
        "High-Value and Frequency Shoppers": 'purple',
        "Balanced Shoppers": 'blue',
        "Moderate-Frequency High-Value Shoppers": 'red'
    }
    
    # Create a consistent cluster_palette using the cluster_colors dictionary
    cluster_palette_rfm = sns.color_palette([cluster_colors_rfm[label] for label in customer_history_df['Segment_rfm']])
    
        
        # Map cluster labels to segment names for RFM
    customer_history_df['Segment_rfm'] = customer_history_df['Cluster_rfm'].map(segment_labels_rfm)
        
        # Create a consistent cluster_palette using the cluster_colors dictionary
    cluster_palette = sns.color_palette([cluster_colors[label] for label in segment_labels_rfm.values()])

    st.header("RFM Analysis Results")

 
    
    import matplotlib.gridspec as gridspec
    # Create a figure and a GridSpec to arrange subplots
    fig = plt.figure(figsize=(12, 18))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
    
    # Subplot 1
    ax1 = plt.subplot(gs[0])
    sns.scatterplot(data=customer_history_df, x='recency_log', y='frequency_log', hue='Segment_rfm', palette=cluster_palette)
    plt.xlabel("Recency Log")
    plt.ylabel("Frequency (Log)")
    ax1.set_title("Recency vs Frequency")
    
    # Subplot 2
    ax2 = plt.subplot(gs[1])
    sns.scatterplot(data=customer_history_df, x='recency_log', y='amount_log', hue='Segment_rfm', palette=cluster_palette)
    plt.xlabel("Recency Log")
    plt.ylabel("Monetary Value (Log)")
    ax2.set_title("Recency vs Monetary Value")
    
    # Subplot 3
    ax3 = plt.subplot(gs[2])
    sns.scatterplot(data=customer_history_df, x='frequency_log', y='amount_log', hue='Segment_rfm', palette=cluster_palette)
    plt.xlabel("Frequency (Log)")
    plt.ylabel("Monetary Value (Log)")
    ax3.set_title("Frequency vs Monetary Value")
    
    # Adjust layout and add to Streamlit
    plt.tight_layout()
    st.pyplot(fig)

    # Display explanations for each segment
    st.subheader("Segment Explanations")
    st.write("""
             
Cluster 0 - Regular Shoppers:
These are customers who shop regularly but not frequently. They have average spending habits.

Cluster 1 - Occasional Shoppers:
Customers in this group shop infrequently and have relatively low spending.

Cluster 2 - High-Value Shoppers:
This cluster consists of customers who shop rarely but spend a significant amount when they do.

Cluster 3 - Balanced Shoppers:
Customers in this category shop regularly, with moderate frequency and spending.

Cluster 4 - Moderate-Frequency High-Value Shoppers:
These customers shop moderately often and have above-average spending habits.""")



    # ... (previous code) ...

if tab_selected == "RFM Clustering":
    # ... (previous RFM Clustering code) ...

    # Display a subset of the DataFrame with recency, frequency, and monetary values
    st.subheader("Subset of Customer Data")
    subset_columns = ['Customer Name', 'recency', 'frequency', 'Value']
    subset_df = customer_history_df[subset_columns]
    st.dataframe(subset_df)

    # Allow the user to input a customer name for quick search
    search_customer = st.text_input("Search for a Customer:", "")

    # Find the closest matches to the searched customer name
    closest_matches = subset_df['Customer Name'][subset_df['Customer Name'].str.contains(re.escape(search_customer), case=False, regex=True)]

    if len(closest_matches) > 0:
        selected_customer = st.selectbox("Select a Customer:", closest_matches.tolist())
        selected_row = subset_df[subset_df['Customer Name'] == selected_customer].iloc[0]

        st.subheader(f"RFM Attributes for {selected_customer}")
        st.write(f"Recency: {selected_row['recency']} days")
        st.write(f"Frequency: {selected_row['frequency']} purchases")
        st.write(f"Monetary Value: {selected_row['Value']} AED")

        # Display the cluster information for the selected customer
        cluster_name = customer_history_df[customer_history_df['Customer Name'] == selected_customer]['Segment_rfm'].iloc[0]
        st.write(f"Cluster: {cluster_name}")

    else:
        st.write("No matching customers found.")

# ... (rest of the code) ...


elif tab_selected == "Purchasing Behavior Clustering":
    # ... (previous Purchasing Behavior Clustering code) ...

# ... (rest of the code) ...



    # Select relevant features for segmentation
    num_features = ['Volume', 'Value', 'Price(AED)']

    # Extract the selected features from the dataset
    X = df1[num_features]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_clusters_pb = 5 
    model_pb = KMeans(n_clusters=n_clusters_pb, random_state=101)
   
    model_pb.fit(X_scaled)

    # Apply K-means clustering with the best K value
      # Update with your desired number of clusters
    
    # labels_pb = model_pb.fit_predict(X_scaled)

    # Add cluster labels to the DataFrame for Purchasing Behavior analysis
    df1['Cluster_pb'] = model_pb.fit_predict(X_scaled)
    
    # Define cluster names for Purchasing Behavior
    



    # Specify the cluster names
    cluster_names = {
0:  'Low-Value, Low-Volume Customers',
1: 'High-Value, High-Volume Customers',
2: 'Moderate-Value, Moderate-Volume Customers',
3: 'Ultra-High-Value, Low-Volume Customers',
4: 'Moderate-Value, Low-to-Moderate-Volume Customers'
}
    
    
    cluster_colors = {
    0: 'blue',
    1: 'green',
    2: 'orange',
    3: 'purple',
    4: 'red'
}
    # Specify the selected cluster numbers
    selected_clusters = list(cluster_names.keys())
    
    
    df1 = df1[df1['Cluster_pb'].isin(selected_clusters)]
    
    cluster_palette = sns.color_palette([cluster_colors[c] for c in cluster_names.keys()])
    
    # Add a new column 'Cluster_Name' to the DataFrame
    df1['Cluster_Name'] = df1['Cluster_pb'].map(cluster_names)
    
   

    # Your data loading and preprocessing code here
    
    # Your clustering code here
    import matplotlib.gridspec as gridspec
    # Create a figure and a GridSpec to arrange subplots
    fig = plt.figure(figsize=(15, 18))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
    
    # Subplot 1
    ax1 = plt.subplot(gs[0])
    sns.scatterplot(data=df1, x='Value', y='Volume', hue='Cluster_Name', palette=cluster_palette)
    plt.xlabel('Value')
    plt.ylabel('Volume')
    ax1.set_title('Value vs Volume')
    
    # Subplot 2
    ax2 = plt.subplot(gs[1])
    sns.scatterplot(data=df1, x='Price(AED)', y='Volume', hue='Cluster_Name', palette=cluster_palette)
    plt.xlabel('Price(AED)')
    plt.ylabel('Volume')
    ax2.set_title('Price vs Volume')
    
    # Subplot 3
    ax3 = plt.subplot(gs[2])
    sns.scatterplot(data=df1, x='Price(AED)', y='Value', hue='Cluster_Name', palette=cluster_palette)
    plt.xlabel('Price(AED)')
    plt.ylabel('Value')
    ax3.set_title('Price vs Value')
    ax3.legend(loc='upper right')
    
    # Adjust layout and add to Streamlit
    plt.tight_layout()
    st.pyplot(fig)
        
    
   

    st.subheader("Segment Explanations")
    st.write("""
Cluster 0: "Low-Value, Low-Volume Customers" - Customers with relatively low purchase values and volumes.

Cluster 1: "High-Value, High-Volume Customers" - Customers making high-value purchases.

Cluster 2: "Moderate-Value, Moderate-Volume Customers" - Customers with moderate purchase values and volumes.

Cluster 3: "Ultra-High-Value, Low-Volume Customers" - Customers making sporadic but very high-value purchases.

Cluster 4: "Moderate-Value, Low-to-Moderate-Volume Customers" - Customers with moderate purchase values and varying volumes.
""")
  


    
    # Display a subset of the DataFrame with cluster information
    st.subheader("Subset of Customer Data")
    subset_columns_pb = ['Customer Name', 'Value', 'Volume', 'Cluster_Name']
    subset_df_pb = df1[subset_columns_pb]
    st.dataframe(subset_df_pb)

    # Allow the user to input a customer name for quick search
    search_customer_pb = st.text_input("Search for a Customer:", "")

    # Find the closest matches to the searched customer name
    closest_matches_pb = subset_df_pb['Customer Name'][subset_df_pb['Customer Name'].str.contains(re.escape(search_customer_pb), case=False, regex=True)]

    if len(closest_matches_pb) > 0:
        selected_customer_pb = st.selectbox("Select a Customer:", closest_matches_pb.tolist())
        selected_row_pb = subset_df_pb[subset_df_pb['Customer Name'] == selected_customer_pb].iloc[0]

        st.subheader(f"Purchasing Behavior Attributes for {selected_customer_pb}")
        st.write(f"Value: {selected_row_pb['Value']} AED")
        st.write(f"Volume: {selected_row_pb['Volume']}")
        st.write(f"Cluster: {selected_row_pb['Cluster_Name']}")

    else:
        st.write("No matching customers found.")

# ... (rest of the code) ...

