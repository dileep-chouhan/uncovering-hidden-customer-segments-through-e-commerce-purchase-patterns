import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
# Number of customers
num_customers = 500
# Generate synthetic customer data
data = {
    'CustomerID': range(1, num_customers + 1),
    'Age': np.random.randint(18, 65, size=num_customers),
    'Income': np.random.randint(20000, 150000, size=num_customers),
    'PurchaseFrequency': np.random.poisson(lam=2, size=num_customers), #average purchases per year
    'AverageOrderValue': np.random.uniform(50, 500, size=num_customers)
}
df = pd.DataFrame(data)
df['TotalSpent'] = df['PurchaseFrequency'] * df['AverageOrderValue']
# --- 2. Data Analysis ---
# Calculate descriptive statistics
print("Descriptive Statistics:")
print(df.describe())
# --- 3. Customer Segmentation using K-Means Clustering ---
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Scale the data for K-Means
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Age', 'Income', 'PurchaseFrequency', 'AverageOrderValue', 'TotalSpent']])
# Determine optimal number of clusters (e.g., using the Elbow method -  simplified here)
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8,6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.savefig('elbow_method.png')
print("Plot saved to elbow_method.png")
# Perform K-Means clustering (using k=3 as an example based on visual inspection of Elbow method)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)
# Analyze cluster characteristics
print("\nCluster Characteristics:")
print(df.groupby('Cluster')[['Age', 'Income', 'PurchaseFrequency', 'AverageOrderValue', 'TotalSpent']].mean())
# --- 4. Visualization ---
# Create a pairplot to visualize relationships between variables and clusters
plt.figure(figsize=(10,8))
sns.pairplot(df, hue='Cluster', vars=['Age', 'Income', 'PurchaseFrequency', 'AverageOrderValue', 'TotalSpent'], diag_kind='kde')
plt.savefig('pairplot_clusters.png')
print("Plot saved to pairplot_clusters.png")
#Further analysis and visualizations can be added here based on the identified clusters.  For example, you could create bar charts showing the distribution of key variables within each cluster.