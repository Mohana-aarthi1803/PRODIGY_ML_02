import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the retail customer dataset from Excel
file_path = 'C:/Users/mohan/Downloads/customer/Online Retail.xlsx'
customer_data = pd.read_excel(file_path)

# Select relevant features for clustering (adjust as needed)
features = ['Quantity', 'UnitPrice', 'InvoiceDate']
X = customer_data[features]

# Drop rows with missing values
X = X.dropna()

# Select the first 200 rows
X_subset = X.head(200)

# Exclude 'InvoiceDate' column for scaling
X_scaled = X_subset.drop(columns=['InvoiceDate'])  # Exclude 'InvoiceDate' from scaling

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_scaled)

# Initialize a range of cluster numbers
min_clusters, max_clusters = 2, 10
cluster_range = range(min_clusters, max_clusters + 1)

# Store inertia values for each cluster number
inertia_values = []

# Find inertia values for different cluster numbers
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot inertia values for different cluster numbers (elbow method)
plt.plot(cluster_range, inertia_values, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters (First 200 rows)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Choose the optimal number of clusters (k) based on the elbow point
optimal_k = 3  # Adjust this based on the elbow method plot

# Apply K-means clustering with the optimal k
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
customer_data_subset = customer_data.head(200)
customer_data_subset['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
# Visualize the clusters as a scatter plot
plt.scatter(X_subset['Quantity'], X_subset['UnitPrice'], c=customer_data_subset['Cluster'], cmap='viridis', alpha=0.5)
plt.xlabel('Quantity')
plt.ylabel('UnitPrice')
plt.title(f'K-means Clustering of Retail Customers (k={optimal_k}, First 200 rows)')

plt.show()
