---
title: "Where people stay - extracting destinations from GPS data"
excerpt: "This notebook showcases some on the extraction of destinations from GPS data carried out during my summer 2017 internship."
categories:
  - demo
tags: 
  - datascience
  - gps
  - mobility
toc: false
---

<style>
    .right {
        float: right;
        width: 200px;
        border: 0px solid #BDBDBD;
        background: #FFF;
        padding: 5px;
        margin:5px;
    }

    figcaption{
    text-align:center;
    }
</style>

<a id='beginning'></a>
Note: This post is a static demonstration of the [destinations extraction project](/projects/). If you prefer running a Jupyter notebook please check out the [GitHub repository](https://github.com/sebastianbertoli/Github-internship_human_mobility).
{: .notice}

# Introduction
This notebook showcases some of the work completed during my summer 2017 internship at the [Bruno Kessler Foundation](https://www.fbk.eu/en/). My responsibility was to find and implement an efficient algorithm that could, using GPS data, tell us where people had stayed for a pre-determined amount of time. So called stop-locations.

To extract stop-locations from GPS data I implemented and tested two algorithms: [ST-DBSCAN](https://www.sciencedirect.com/science/article/pii/S0169023X06000218) and an algorithm proposed by Kentaro and Toyama in [[1]](#hariharan2004). The focus of this notebook will be the latter.

## Outline

This notebook is structured as follows. First, we load and explore dataset I prepared for this experiment. Second, we process it and extract its stop locations. Third, we proceed with clustering the stop locations into so-called destinations (more on this later) In short:

1. [Exploring the dataset](#eda)
1. [Extracting the users' stop locations](#extract_stops)
1. [Clustering the locations into destinations](#cluster_stops)

Note: Most functions and plots are loaded from the accompanying `lachesis.py` and `plotly_helpers.py` files to avoid overloading this notebook with code. They can be found in the project's [repository](https://github.com/sebastianbertoli/Github-internship_human_mobility). [Final thoughts](#finalthoughts), [acknowledgments](#acknowledgments) and [references](#references) can be found at the bottom.
{: .notice}

<a id='eda'></a>

# Exploring the Data

The proliferation of smartphones with GPS sensors has allowed to capture people's movements in regular time intervals and at large scale. Once the GPS data is collected, it can be further processed for research purposes. The unprocessed data typically consists of a timestamp, some type of user or device identifier and longitude-latitude coordinates.

For the experiments we will use a random sample of the [T-Drive trajectory data sample](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/). The sample contains one week's worth of GPS data collected from 100 taxis driving in Bejing.

Note: Before running the experiments please make sure that you have all the libraries listed in `requirements.txt` installed.
{: .notice}

Without further adue let us load and explore the data!

```python
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly
from plotly_helpers import *  # Plot specifications
from lachesis import *  # Stop detection implementations
# Various notebook settings
plotly.offline.init_notebook_mode()
%load_ext autoreload
%autoreload 2
%matplotlib inline
mpl.rcParams['figure.dpi'] = 240
plt.style.use('ggplot')
```

```python
df = (pd.read_csv("data/df_sample.csv", parse_dates=["timestamp"])
      .sort_values(['user_id', 'timestamp']))
print(df.iloc[:3,:], 
      "\n\nNumber records:", df.shape[0],
      "\nNumber users:", len(df["user_id"].unique()))
```

           user_id           timestamp  longitude  latitude
    48013      165 2008-02-02 13:46:16  116.49889  39.92208
    48014      165 2008-02-02 13:56:16  116.48131  39.92130
    48015      165 2008-02-02 14:06:15  116.42837  39.90675 
    
    Number records: 111712 
    Number users: 100

As we can see our sample dataset contains of 111,712 records coming from 100 different users. Each record has a `user_id` identifying the user, a `timestamp` and a `longitude` as well as a `latitude` value. 

One important property of GPS data is the time elapsed between subsequent observations denoted as delta t. Let's quickly calculate it and plot its distribution.

```python
df_stats = pd.DataFrame()
df_stats['delta_t'] = (df.groupby('user_id')['timestamp']
                       .transform(lambda x: x.diff()) / np.timedelta64(1,'s'))
delta_t_plot = (df_stats[df_stats['delta_t'] < df_stats['delta_t']
                .quantile(.96)]
                .hist(bins=50, figsize=(6,2), alpha=.9, color="blue"))
```

![png](/assets/images/posts/2018-01-11-human-mobility/output_9_0.png)

We observe that for most records (96th percentile) the time lapsed between subsequent recordings is within 310 seconds. This is sufficient to run our stop-detection algorithms. 

Now let us get a better idea of the data by plotting a random sample on a map. We use the popular [Plotly library](https://plot.ly) for this. The plot is fully interactive so feel free explore the data on the map. 

```python
fig_datsample = plot_datasample(df.sample(df.shape[0]//10, random_state=10))
plotly.offline.iplot(fig_datsample, filename='fig_datsample')
```

![jpg](/assets/images/posts/2018-01-11-human-mobility/fig_datsample.jpg)

Despite having only plotted 10% of the data we still get a good idea of the area covered by the dataset. Next, let us look at a handful of data points of user 4813. 

```python
fig_oneuser = plot_one_user(df[df['user_id'] == 4813].iloc[109:115,:])
plotly.offline.iplot(fig_oneuser, filename='fig_oneuser')
```

![jpg](/assets/images/posts/2018-01-11-human-mobility/fig_oneuser.jpg)

In the figure above we have plotted six data points. Let us first focus on the three points in the upper right corner east of the *China Foreign Affairs University*. If you zooms-in on those points, you will thee that the user has stayed in close proximity of that location from `12:41` to `12:57`. Subsequently, the users moves south (`12:58`) to *Outer Fuchengmen Street* (`13:00`) and moves west (`13:02`) until he disappears from the frame.

The three points in the upper right corner should be a stop location. In other words, a location where a user (in this case a taxi) has stayed for some time. Unfortunately, the dataset does not yet have this information baked-in. That is why we will add this information in the next section.

<a id='extract_stops'></a>

# Extract the users' stop-locations

In this section we briefly explain the stop location extraction algorithm from [[1]](#hariharan2004). After that we extract all the stop locations from the sample dataset. Let us begin with some definitions. 

## Definitions

There are three basic concepts that are important for us: the position, the stop location and the destination.

<span style="color:blue">&#9679;</span> **Position**: tells us where a user was at a certain point in time. It is just an observation from the dataset consisting of a *timestamp*, a *user identifier* as well as *longitude* and *latitude* coordinates.

<span style="color:orange">&#9679;</span> **Stop location**: tells us where and when a user has stopped (stayed). It consists of a *timestamp*, a *user identifier* as well as *longitude* and *latitude* coordinates. It also has two additional variables: *t_start* and *t_end* indicating the start- and end-time of the stop.

<a id='destination_definition'></a><span style="color:green">&#9679;</span> **Destination**: aggregates stop locations in close proximity of each other and tells us where and how often a user has stopped there.  Similar to the position, the destination consists of a *timestamp*, a *user identifier* as well as *latitude* and *longitude* coordinates. In addition, the *visitation count* denotes the number of stops at a particular destination and *cluster_assignment* identifies the destination itself.

{% include figure image_path="/assets/images/posts/2018-01-11-human-mobility/definitions.jpg" alt="Figure 1: Definitions of the key concepts." caption="Figure 1: Definitions of the key concepts." %}

## The stop location algorithm explained

Below we have pictured a fictional user's positions with associated timestamps. The user is moving from left to right starting at 8:24. Between 8:26 and 8:46 the user moves less than 50 meters in 20 minutes. Thus, the algorithm detects that the user has stopped. It picks the [medoid](https://en.wikipedia.org/wiki/Medoid) (in orange) of those three points as the stop location (8:36). The start and end-times of the stop location are 8:26 and 8:46 respectively.

{% include figure image_path="/assets/images/posts/2018-01-11-human-mobility/stoplocation_explanation.jpg" alt="Figure 2: The stop detection algorithm explained." caption="Figure 2: The stop detection algorithm explained." %}

The algorithm has two parameters which are dependent on the application.

**Roaming distance**: defines how far a user is allowed to roam to still be conisdered part of the stop location (dashed orange line).

**Stay duration**: specifies the minimum amount of time a user has to stay within the roaming distance for his position to be considered as being part of a particular stop location.

## Extracting stop locations from the sample data

In the next cell we extract the stop locations from the sample dataset. We set the parameters `roaming_distance` to 50 meters and the `minimum_stay` parameter to 10 minutes.

I implemented the algorithm with parallel processing in mind for further speed gains. Therefore, there is a third parameter called `number_jobs` which specifies how many processors to use for parallel processing. For this example we set it to two. If necessary change it in accordance to your speed requirements and hardware. See the [joblib documentation](https://pythonhosted.org/joblib/parallel.html#parallel-reference-documentation) for details.

Now, let us run the cell below to extract the stop locations for each user! It should take less than 30 seconds to complete.

```python
# Parameters
roaming_distance = meters2degrees(50) # 50 meters converted to degrees
minimum_stay = 10  # minutes
number_jobs = 2 # number of parallel jobs

# Set index
df = df.reset_index().set_index(["user_id", "timestamp"])

# Call helper-function to process entire df in one go
df_stops = process_data(df=df, 
                        roam_dist=roaming_distance, 
                        min_stay=minimum_stay, 
                        n_jobs=number_jobs,
                        print_output="notebook")
df_stops = pd.concat(df_stops)

# Only keep users with more than one stop
df_stops = (df_stops
    .groupby("user_id").filter(lambda x: len(x) > 1)
    .set_index(["user_id"]))

# Preview data
print("Number of stop locations: {}".format(df_stops.shape[0]))
df_stops.iloc[:3,:]
```

    Processing user 100 of 100.
    Number of stop locations: 2557

|timestamp|latitude|longitude|t_start|t_end|user_id|
|--- |--- |--- |--- |--- |--- |--- |--- |--- |
|2008-02-08 12:07:13|40.23100|116.69343|2008-02-08 10:57:14|2008-02-08 13:17:13|165|
|2008-02-04 01:56:03|40.17047|116.73509|2008-02-03 23:06:04|2008-02-04 09:46:00|165|
|2008-02-02 14:56:15|39.95174|116.44591|2008-02-02 14:56:15|2008-02-02 15:26:15|165|

Congratulations! We have just successfully extracted 2557 stop locations from the data set. As you can see from the output above each stop location has now a start time and end time (`t_start`, `t_end`). Let us plot the stop locations on a map.

```python
fig_stops = plot_stops(df_stops.reset_index())
plotly.offline.iplot(fig_stops, filename='fig_stops')
```
![jpg](/assets/images/posts/2018-01-11-human-mobility/fig_stops.jpg)

Next, we will aggregate the stop locations into destinations.

 <a id='cluster_stops'></a>

# Aggregate stop locations into destinations

## Why do we need destinations?

Recall that we defined a [destination](#destination_definition) as the aggregation of one or several stop locations that are in close proximity to each other. Basically, we want to move from a stop locations representation to a destinations representation (figure 3).

<a id='figure_3'></a>

{% include figure image_path="/assets/images/posts/2018-01-11-human-mobility/locationdestination.jpg" alt="Figure 3: From stop locations to destinations." caption="Figure 3: From stop locations to destinations." %}

<figure class="right">
    <img src="/assets/images/posts/2018-01-11-human-mobility/stop_locations.jpg"/> 
</figure>

To appreciate why this is useful, let us have a look at the figure on the right where we have plotted some stop locations in orange (from a different dataset). The stop locations appear to form small clusters: one bigger cluster in proximity of *Building 7* on the left and two other clusters in proximity of *Building 1* and *Building 2* respectively. Finally, there are also two stop locations on *Malcolm Boulevard* at the bottom.

What we are actually interested in is the *destination* where a user has stopped and not necessarily the exact *GPS position* recorded. For instance, the group of stop locations surrounding *Building 7* should be regarded as one destination. Similarly, those surrounding *Building 1* and *Building 2* respectively. In other words, we need to cluster or aggregate the stop locations into destinations.

To aggregate the stop locations into destinations we use [Scipy's hierarchical clustering functions](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html). For our application there are two important parameters: The [linkage parameter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage) defines how the clusters are formed. The [distance parameter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster) specifies within what spatial distance stop locations are to be considered of the same cluster. Let us briefly look at two linkage methods in practice.

Figure 4 shows the difference between using a complete linkage method and a centroid linkage method using the same distance parameter. For our purpose complete linkage forms too many small clusters. In contrast, centroid linkage seems to be just right forming approximately one cluster for each building. Of course, these parameters will have to be set based on the application at hand.

{% include figure image_path="/assets/images/posts/2018-01-11-human-mobility/destinations_composite.jpg" alt="Figure 4: A comparison of linkage methods for clustering GPS positions." caption="Figure 4: A comparison of linkage methods for clustering GPS positions." %}

## Running the clustering algorithm on the stop locations data

In the next cell we finally cluster the stop locations into destinations. We set the parameter `linkage_method` to 'centroid' and the `distance` threshold to 100 meters.

```python
# Clustering parameters
linkage_method = 'centroid'
distance = meters2degrees(100)

# Cluster stoplocations on a per user basis
df_clusters = (df_stops.groupby('user_id')
              .apply(lambda x: 
                     cluster_stoplocations(x, 'centroid', distance))
              .reset_index())
# Preview results
df_clusters.iloc[:3,:]
```

|user_id|timestamp|latitude|longitude|t_start|t_end|cluster_a...|
|--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |
|165|2008-02-08 12:07:13|40.23100|116.69343|2008-02-08 10:57:14|2008-02-08 13:17:13|1|
|165|2008-02-04 01:56:03|40.17047|116.73509|2008-02-03 23:06:04|2008-02-04 09:46:00|2|
|165|2008-02-02 14:56:15|39.95174|116.44591|2008-02-02 14:56:15|2008-02-02 15:26:15|10|

Great, each stop locations has been assigned to a destination denoted by the new column `cluster_assignment`.

### Get medoid for each destination

Now that we have assigned each stop location to a destination we want to have only one point representing each destination. We saw briefly this in [figure 3](#figure_3) where one point per cluster is marked with a `M`. 

To have one representative point per cluster, we calculate the [medoid](https://en.wikipedia.org/wiki/Medoid) of the stop locations within each destination. We also calculate the number of stop locations at each destination, which is useful for ranking destinations based on visitation counts.

```python
# Get medoid of each destination (cluster)
df_clustermedoids = (df_clusters.groupby('user_id')
    .apply(lambda x: get_clustermedoids(x))
    .reset_index(drop=True))

# Compute stop counts at each destination
df_clustersizes = (df_clusters
                   .groupby(['user_id', 'cluster_assignment'])
                   .apply(lambda x: len(x))
                   .reset_index(name='count'))

# Merge medoids and counts
df_destinations = pd.merge(df_clustermedoids.loc[:,['user_id', 'timestamp',
                                                 'latitude', 'longitude',
                                                 'cluster_assignment']], 
                             df_clustersizes, 
                             on=['user_id', 'cluster_assignment'], 
                             how='left')
# Export and preview data
df_destinations.iloc[:3,:]
```

|user_id|timestamp|latitude|longitude|cluster_assignment|count|
|--- |--- |--- |--- |--- |--- |--- |--- |--- |
|165|2008-02-08 12:07:13|40.23100|116.69343|1|5|0|
|165|2008-02-06 00:41:55|40.17044|116.73519|2|7|1|
|165|2008-02-05 22:11:56|40.16936|116.73150|3|1|2|

Congratulations, we now have a new dataset `df_destinations` containing the destinations of each user and their respective visitation frequency. Let us plot our final result on a map.

```python
fig_destinations = plot_destinations(df_destinations)
plotly.offline.iplot(fig_destinations, filename='fig_destinations')
```

![jpg](/assets/images/posts/2018-01-11-human-mobility/fig_destinations.jpg)

On the map you can appreciate the main destinations on a per-user basis the colour intensity and the size indicate the visitation frequency (number of stop locations) at each destination.

## Bonus - destinations across all users

Instead of clustering the stop locations into destinations for each user separately we can also do that for all 100 users combined. This allows us to see the top destinations across the city. Therefore, let us redo the clustering on all stop locations combined and then plot the results on a map. The next cell will take approximately 15 seconds to run.

```python
# Clustering parameters
linkage_method = 'centroid'
distance = meters2degrees(100)

# Cluster stoplocations on a per user basis
df_clusters_all = (cluster_stoplocations(df_stops, 'centroid', distance)
                   .reset_index())

# Get medoid of each destination (cluster)
df_clustermedoids_all = (get_clustermedoids(df_clusters_all)
                         .reset_index(drop=True))

# Compute stop counts at each destination
df_clustersizes_all = (df_clusters_all
                   .groupby(['cluster_assignment'])
                   .apply(lambda x: len(x))
                   .reset_index(name='count'))

# Merge medoids and counts
temp_cols = ['user_id', 'timestamp','latitude', 'longitude',
             'cluster_assignment']
df_destinations_all = pd.merge(df_clustermedoids_all.loc[:, temp_cols], 
                             df_clustersizes_all, 
                             on=['cluster_assignment'], 
                             how='left')
```

```python
fig_destinations_all = plot_destinations(
    df_destinations_all[df_destinations_all['count'] > 1])
plotly.offline.iplot(fig_destinations_all, filename='fig_destinations_all')
```

![jpg](/assets/images/posts/2018-01-11-human-mobility/fig_destinations_all.jpg)

Well done! We now have plotted the most popular taxi destinations in Bejing based on visitation count. 

<a id='finalthoughts'></a>

# Final thoughts

To sum up, we have loaded a GPS dataset, extracted the stop locations and clustered them into destinations. Now that we have this information we can do so much more. For instance we could calculate the [radius of gyration](https://en.wikipedia.org/wiki/Individual_mobility#Characteristics) of each user.

```python
# Cluster stoplocations on a per user basis
df_rgyration = (df_destinations
                .loc[:,['user_id', 'longitude', 'latitude', 'count']]
                .groupby('user_id')
                .apply(lambda x: rgiration_at_k(x, k=None))
                .reset_index())
df_rgyration.columns = ['user_id', 'radius_gyration']
df_rgyration.iloc[:3,:]
```

|user_id|radius_gyration|
|--- |--- |--- |--- |--- |
|165|14371.809524|0|
|185|23379.968301|1|
|685|35287.929649|2|

Using the radius of gyration measure we could now start characterising the mobility patterns of our users more precisely... but let that be the topic for a new notebook. :-)

**Thanks for reading this far**. If you found this notebook useful feel free to [follow me on the web](http://www.sebastianbertoli.net/). Comments and feedback are always appreciated!{: .notice}

<a id='acknowledgments'></a>

# Acknowledgments

My thanks go to Marco de Nadai and Lorenzo Lucchini for their insights and guidance as well as to the staff of the Bruno Kessler foundation for arranging the internship.

<a id='references'></a>

# References

<a id='hariharan2004'></a> [1] Hariharan R., Toyama K. (2004) [Project Lachesis: Parsing and Modeling Location Histories.](https://link.springer.com/chapter/10.1007/978-3-540-30231-5_8#citeas) In: Egenhofer M.J., Freksa C., Miller H.J. (eds) Geographic Information Science. GIScience 2004. Lecture Notes in Computer Science, vol 3234. Springer, Berlin, Heidelberg

[Back to top](#beginning)
