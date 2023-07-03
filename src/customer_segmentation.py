# Import Libraries
from sklearn.cluster import KMeans
import plotly.express as px
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Import Cleaned Dataset
df = pd.read_csv(r"D:\Personal Projects\customer_segmentation\data\processed_data.csv")

# Modellin K-Means with K=6
kmeans = KMeans(n_clusters=6)    
kmeans.fit(df)
df['Label']=kmeans.labels_

# Cluster analysis
def plot_cluster(n_cluster):
    cluster = df[df['Label']==n_cluster].loc[:,:"Recency"]

    fig = px.line_polar(cluster,
                        r = cluster.mean().tolist(),
                        theta = cluster.columns.tolist(),
                        line_close = True)
    fig.update_layout(
        title="Cluster {}".format(n_cluster),
    )
    fig.show()

for i in range(0,6):
    plot_cluster(i)