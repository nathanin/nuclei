# Scripts for working up the handcrafted-features

We'll work with data where the main tables are separated from the labels, using the index column, and case-id columns to translate between features and labels.

For each feature type we'll do two things: filtering based on variance and correlations, then unsupervised heriarchical clustering on the tile-dimension and feature-dimensions. 