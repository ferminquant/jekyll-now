---
layout: post
title: Pattern Recognition with Autoencoders
---

For this post, I use the [Amazon Commerce reviews set Data Set](http://archive.ics.uci.edu/ml/datasets/Amazon+Commerce+reviews+set#). I will play around with autoencoders for pattern recognition.

<!-- MarkdownTOC autolink="true" bracket="round" depth="0" style="unordered" indent="  " autoanchor="false" -->

- [What is this data set?](#what-is-this-data-set)
- [Loading the data into R](#loading-the-data-into-r)

<!-- /MarkdownTOC -->

# What is this data set?

According to the uploader from the Machine Learning Repository, it is a data set of 50 reviewers with 30 reviews each, hence 1,500 rows. It has 10,000 columns, which represent the reviews written separated into punctuations used, digits used, frequency of words, etc. I couldn't find an exhaustive list, but it really isn't necessary.

The purpose of the dataset is to try out Writeprint, which is more or less like identifying the writing fingerprint of each author. In the end it is a classification problem, but with 50 different classes.

# Loading the data into R

Here is the code to load the ARFF file into R.

```r
library(foreign)
data = read.arff("Amazon_initial_50_30_10000.arff")
```

However, it will fail miserably with the cryptic error somewhere along the lines of: "Invalid attribute specification line 1 appears to contain an embedded nul."
To solve this refer to the second answer by Chloe [here](http://stackoverflow.com/questions/9951839/reading-arff-file-in-r). The problem is there are multiple attributes with the same name, a very simple solution is provided in the link, after which the read.arff function will work. I modified the values with the text editor Sublime Text in Ubuntu, in case you are having problems with yours.

You can also use the H2O function, which reads the ARFF file directly into an H2O data frame:

```r
hdata = h2o.importFile(path = paste(getwd(),"Amazon_initial_50_30_10000.arff",sep="/"),
                       destination_frame = "hdata")
```

