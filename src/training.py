# Import Libraries
from sklearn.cluster import KMeans
import plotly.express as px
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# Import Cleaned Dataset
df_scaled = pd.read_csv(os.getcwd() + '\data\standardized_data.csv')
df_cleaned = pd.read_csv(os.getcwd() + '\data\cleaned_data.csv')

# K-Means with K=6
kmeans = KMeans(n_clusters=6, random_state=0)    
preds = kmeans.fit_predict(df_scaled)
df_scaled['Label'] = preds

# Save the model
pickle.dump(preds, open(os.getcwd() + '\models\clustering_kmeans.pkl', 'wb'))

# Visualize cluster charateristics
def plot_cluster(n_cluster):
    cluster = df_scaled[df_scaled['Label']==n_cluster].loc[:,:"Recency"]

    fig = px.line_polar(cluster,
                        r = cluster.mean().tolist(),
                        theta = cluster.columns.tolist(),
                        line_close = True)
    fig.update_layout(
        title="Cluster {}".format(n_cluster+1),
    )
    fig.show()

for i in range(0,6):
    plot_cluster(i)

# add clusters label column to the RMF df
df_cleaned['Label'] = kmeans.labels_

# change the cluster label order
change_order = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6}
df_cleaned['Label'] = df_cleaned['Label'].replace(change_order)

# saving cluster df to csv
output_directory = os.getcwd() + "\\data\clusters\\"
for label in range(1, 7):
    cluster = df_cleaned[df_cleaned['Label'] == label]
    file_path = f"{output_directory}cluster_{label}.csv"
    cluster.to_csv(file_path, index=False)
    print(f"Saved cluster {label} to {file_path}")