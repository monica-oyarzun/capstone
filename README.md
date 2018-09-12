# Inside Out – Using Unsupervised Machine Learning to Cluster Daily Emotions Tracking Data 

This repository is where I am keeping my notebooks and data analysis for my final capstone project at Galvanize. It is a work in progress, and I will be posting updates as I continue to try new models and find new insights from my dataset.

To assist with navigating this repository:
-	Data cleaning and transformations are completed in src.pre_processing.py
-	Final modeling code is in src.model.py
-	Final modeling pipeline and resulting visualizations can be viewed in kMeans_fullData_9clusters.ipynb
-	Additional modeling pipeline, for 6 clusters with 10 components, can be viewed in kMeans_fullData_6clusters.ipynb
-	Because the data is private, it is not included in this repository

## Data and Context:

This data comes from EmoTrak, an emotions-tracking app developed by Eve Ekman, PhD, who is a contemplative social scientist and teacher in the field of emotional awareness and burnout prevention. She developed this app to explore measurements of burnout and daily emotions, applied it first to a pilot study across 100 medical residents, and then expanded this to reach a larger, more varied population. All participants were reporting high levels of stress and burnout.

The data I am working with is from this larger population, and consists of responses from 347 individuals. The median timeframe for participation per individual was 14 continuous days, and the median entries per participant during that timeframe was 21, resulting in 1.3 responses per day. Although participants came into the study reporting high levels of stress and burnout, the predominant emotion that was tracked on a daily basis was enjoyment, and supports similar findings from Dr. Ekman’s pilot study. As stated in her pilot study report, this indicates that “our moment to moment experience can be quite distinct from our ‘global’ assessment of how we are doing.” 

## Preliminary EDA

Initial data analysis showed that this population tracked over 50% of their daily emotions as positive emotions of enjoyment, resulting primarily from triggers of wellbeing, work satisfaction, and social support. These findings can be seen in the counts of the plots below. 

![genEmotions](https://github.com/monica-oyarzun/inside-out/blob/master/pictures/genEmotions.png?raw=true)

![genTriggers](https://github.com/monica-oyarzun/inside-out/blob/master/pictures/genTriggers.png?raw=true)

![genEmotions_genTriggers](https://github.com/monica-oyarzun/inside-out/blob/master/pictures/genEmotions_genTriggers.png?raw=true)

## Clustering Model

I hoped to use the data to try to predict burnout rates, but in the end it didn’t present a clear target for any kind of predictions. I opted to explore unsupervised learning methods instead, and used kMeans clustering to categorize the data points. The data is comprised of predominantly categorical features, and after dummifying these to prepare for a clustering model, I had 98 columns. I performed Principal Component Analysis to reduce this to 8, and reviewed various numbers of clusters to compare silhouette scores. The silhouette scores offer a representation of how well the defined clusters belong together, with values near 1 showing clearly definitive clusters and values near 0 showing points that fall along the boundary of a neighboring cluster. Negative values represent points that were clustered incorrectly.

The silhouette graphs for varied n_clusters, with 8 principal components, are below. I chose 9 clusters, which is the first image pictured below, because that model reflected the highest average silhouette score at .381 and clusters that all form points beyond that average. Additionally, those clusters had the least values extending into the negative.

![9_clusters](https://github.com/monica-oyarzun/inside-out/blob/master/pictures/9_clusters.png?raw=true)
![8_clusters](https://github.com/monica-oyarzun/inside-out/blob/master/pictures/8_clusters.png?raw=true)
![5_clusters](https://github.com/monica-oyarzun/inside-out/blob/master/pictures/5_clusters.png?raw=true)


## Clustering Results

Results were clustered based on the primary emotions of anger, fear, enjoyment and sadness. Enjoyment has 4 separate clusters, that are split by the top 3 enjoyment triggers of wellbeing, social support, and work satisfaction. Fear and Anger were split into two clusters each, divided by social stress and work stress. And sadness made up the 9th cluster all on its own. These clusters make sense and support my findings from my initial exploratory analysis. Below are details for each cluster:

![Cluster Results](https://github.com/monica-oyarzun/capstone/blob/master/pictures/cluster_results.png?raw=true)
