def create_cluster_stats_plot(gdf):
    import plotly.express as px
    return px.box(
        gdf,
        x="cluster",
        y="population_density",
        color="cluster",
        title="Population Density by Cluster"
    )

def create_dendrogram(gdf, features):
    from scipy.cluster.hierarchy import linkage, dendrogram
    import plotly.figure_factory as ff
    Z = linkage(gdf[features], 'ward')
    fig = ff.create_dendrogram(Z, orientation='left', labels=gdf['cluster'])
    return fig
