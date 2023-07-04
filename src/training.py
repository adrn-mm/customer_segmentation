# Import Libraries
from sklearn.cluster import KMeans
import plotly.express as px
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# Import Cleaned Dataset
df = pd.read_csv(os.getcwd() + '\data\processed_data.csv')

# K-Means with K=6
kmeans = KMeans(n_clusters=6, random_state=0)    
preds = kmeans.fit_predict(df)
df['Label'] = preds

# Save the model
pickle.dump(preds, open(os.getcwd() + '\models\clustering_kmeans.pkl', 'wb'))

# Visualize cluster charateristics
def plot_cluster(n_cluster):
    cluster = df[df['Label']==n_cluster].loc[:,:"Recency"]

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