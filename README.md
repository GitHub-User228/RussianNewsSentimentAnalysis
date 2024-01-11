# Russian News Sentiment Analysis
ITMO University R&D Project

This repository contains source code used for completing R&D project.

## R&D Project Description 

* <strong>Topic</strong>: the development of news sentiment detection meta model for sensitive content detection in Russian news feeds
* <strong>Problem</strong>: there is a turbulence in economics, politics and social fields, so that some news feeds contain sensitive content for people
* <strong>Object</strong>: textual news from news feeds
* <strong>Subject</strong>: emotional reaction of newsreaders to textual news from news feeds
* <strong>Hypothesis</strong>: if to combine recent techniques of building a meta model on the top of embedding model, the quality of the resulting model might be higher
* <strong>Goal</strong>: to develop sensitive news detection meta model by combining recent techniques to build a meta (hybrid) model on the top of embedding model to detect sentiments in a text

<strong>Major tasks</strong>:
1) Collecting information about recent techniques used to build a hybrid (meta) model for sentiments detection in a text (articles since 2020)
2) Analysing and comparing techniques and models used to build a meta model
3) Collecting, preprocessing and analysing text data
4) Building and training the final model on collected data by combining models and techniques found as a result of Task 1
5) Validating the final meta model

This repository covers Task 4

Short overview of the results and the best meta model can be found in the presentation file

In short, this is the best configuration of the meta model among considered: </br>

![The best meta model](assets/images/best_meta_model.png)

