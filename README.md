[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/osI6zUIF)

# Explainable AI Assignment 2 - Model Explanations

In this assignment, you are challenged to explain a model. For this, you will research exisiting approaches and apply them to your model and interpret the results.

## General Information Submission

For the intermediate submission, please enter the group and dataset information. Coding is not yet necessary.

**Team Name:** KEPL'er

**Group Members**

| Student ID    | First Name  | Last Name      | E-Mail                  | Workload [%] |
| --------------|-------------|----------------|-------------------------|--------------|
| xxxxxxxx      | xxxxx        | xxxxx         |xxxxxxxx  |25%           |
| xxxxxxxx      | Paul        | Dobner-Dobenau |xxxxxxxx  |25%           |
| xxxxxxxx      | Maximilian  | Hageneder      |xxxxxxxx  |25%           |
| xxxxxxxx      | Kilian      | Truong         |xxxxxxxx|25%           |

**Dataset**

For this assignment we are going to analyze the [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) dataset.

We are going to use the [TabNet](https://arxiv.org/pdf/1908.07442)-model on it and analyze what features the model finds important and how it compares to other approaches.

## Final Submission

The submission is done with this repository. Make to push your code until the deadline.

The repository has to include the implementations of the picked approaches and the filled out report in this README.

* Sending us an email with the code is not necessary.
* Update the *environment.yml* file if you need additional libraries, otherwise the code is not executeable.
* Save your final executed notebook(s) as html (File > Download as > HTML) and add them to your repository.

## Development Environment

Checkout this repo and change into the folder:

```
git clone https://github.com/jku-icg-classroom/xai_model_explanation_2024-<GROUP_NAME>.git
cd xai_model_explanation_2024-<GROUP_NAME>
```

Load the conda environment from the shared `environment.yml` file:

```
conda env create -f environment.yml
conda activate xai_model_explanation
```

In case the environment throws an error, you can try this minimal environment:

```
conda env create -f minimal_environment.yml
conda activate xai_model_explanation
```

This method is more reliable across different systems, but it may produce varying results when training the model from scratch. However, the pre-trained model should work consistently in both environments and yield the same results.

> Hint: For more information on Anaconda and enviroments take a look at the README in our [tutorial repository](https://github.com/JKU-ICG/python-visualization-tutorial).

Then launch Jupyter Lab:

```
jupyter lab
```

Alternatively, you can also work with [binder](https://mybinder.org/), [deepnote](https://deepnote.com/), [colab](https://colab.research.google.com/), or any other service as long as the notebook runs in the standard Jupyter environment.

## Report

### Model & Data

* Which model are you going to explain? What does it do? On which data is it used?
* From where did you get the model and the data used?
* Describe the model.

We are going to explain the [TabNet](https://arxiv.org/pdf/1908.07442)-model, an interpretable deep learning architecture for tabular data. It uses sequential attention to select the best features in multiple decision steps. It can perform both classification as well as regression tasks. One of its advantages is that it requires no prior normalization of the feature values for training.

Its interpretability is provided by its attention masks, which allow the user to analyze what features were most influential in the prediction and in which decision steps they were evaluated (concrete details in the section ["Explainability Approaches"](#explainability-approaches) and in the notebook).

We wish to test its explanation-capabilities by comparing it to the general explanation techniques we learned in the lecture. To do so, we applied it to the [Rain in Australia](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) dataset, which comprises about 10 years of daily weather observations from several locations across Australia. It consists of 22 features describing meteorological measurements taken at a weather station at the given location at the given date. The goal is to use these features to predict if it rained on the following day.

### Explainability Approaches

Here we describe our approaches to explain the model which are implemented in the notebook.
The notebook contains the code and plots while the report here is for the interpretation of the results.

#### Approach 1: Permutation Feature Importance

* Briefly summarize the approach.

Permutation Feature Importance calculates the importance of each feature by randomly shuffling the corresponding feature values around in the whole dataset and then calculating the prediction error. The importance score is then calculated as the difference or the ratio to the true prediction error. Features, for which this method causes a high increase in error are deemed as important; those for which the error barely changes are deemed as unimportant.

* Categorize this explainability approach according to the criteria by Hohman et al.

  * WHY?
    * Interpretability & Explainability
    * Debugging & Improving Models
  * WHO?
    * Model Developers & Builders
  * WHAT?
    * Learned Model Parameters
    * Aggregated Information
  * HOW?
    * Algorithms for Attribution & Feature Visualization
  * WHEN?
    * After Training
  * WHERE?
    * Tabular Data

* Interpret the results here. How does it help to explain your model?

The TabNet-model has its own method of calculating feature importance by aggregating the attention masks and calculating the ratio of attention assigned to each feature to the total attention. The Permutation method allowed us to verify and confirm those importances. The model seems to have good intrinsic interpretability.

As for the results themselves, the first takeaway is that the humidity at 03:00PM in the afternoon is by far the most important feature for the model. This aligns with scientific reality, as the probability of rain is directly dependent on the humidity of the air. It is a similar case for wind and air pressure, as differences in air pressure cause strong wind, which encourages cloud formation.
Another relation can be observed where the model mostly prefers measurements in the afternoon over those in the morning for next day prediction.

#### Approach 2: Partial Dependence Plots

* Briefly summarize the approach.

Partial Dependence Plots show the relationship between a feature and the target. First, all the distinct feature values in the dataset are determined. Then for each value, it is set for the entire dataset for that feature and the average prediction probability is calculated.
This is repeated for every distinct value. At the end you get an approximation of how the prediction changes for each value of that feature.

* Categorize this explainability approach according to the criteria by Hohman et al.

  * WHY?
    * Interpretability & Explainability
    * Debugging & Improving Models
  * WHO?
    * Model Developers & Builders
  * WHAT?
    * Learned Model Parameters
    * Aggregated Information
  * HOW?
    * Algorithms for Attribution & Feature Visualization
  * WHEN?
    * After Training
  * WHERE?
    * Tabular Data

* Interpret the results here. How does it help to explain your model?

This gives us more insight in how the prediction is affected by the feature values. The most notable feature
is again Humidity3PM, which has a near linear relation to the target. This again lines up with scientific consensus, where at 0% humidity rain is not possible and at 100% rain has to fall. Generally, it can be observed that the features with low importance have little variance on the y-axis.

Overlaying the plots with a histogram shows that most of the features have pretty evenly distributed values. One notable outlier is Rainfall, which measures the rainfall at a given day in millimeters. Apparently, there was one day in the last ten years, where Australia received catastrophic rainfall at 367 mm.

#### Approach 3: Attention Masks

* Briefly summarize the approach.

TabNet calculates attention masks at each decision step that depict the attention placed on each feature for every instance. These can then be aggregated to visualize the average attention for each feature over each decision step or throughout all of them combined.

* Categorize this explainability approach according to the criteria by Hohman et al.

  * WHY?
    * Interpretability & Explainability
    * Debugging & Improving Models
  * WHO?
    * Model Developers & Builders
    * Model Users
  * WHAT?
    * Learned Model Parameters
    * Aggregated Information
  * HOW?
    * Instance-based Analysis & Exploration
    * Algorithms for Attribution & Feature Visualization
  * WHEN?
    * During Training
    * After Training
  * WHERE?
    * Tabular Data

* Interpret the results here. How does it help to explain your model?

##### Per Instance Masks

The per instance attention masks show how the model assigns importance to different features not only in the decision steps but also in the individual instances. In the first two steps (mask 0 and mask 1) the *important* features vary a lot between the instances, while in the later steps the masks focus more on the same feature over all instances. This might indicate that the model first tries to find the most important features for the individual instances and then refines the prediction by focusing on the same features for all instances.

##### Aggregated Masks

In the aggregated masks visualization we can see how the model assigns attention to the features at each decision step.

**Step 1:**  
*WindDir3pm*, *WindSpeed3pm*, and *Humidity3pm* are the key features, indicating the model starts with analyzing **current 3pm conditions**.

**Step 2:**  
*MinTemp*, *Rainfall*, *WindDir9pm*, and *Humidity3pm* become relevant, suggesting the model incorporates **overnight wind direction** and **daily rainfall** alongside continuing to assess humidity.

**Step 3:**  
*Pressure3pm* dominates, reflecting the importance of **pressure dynamics** for understanding broader weather patterns.

**Step 4:**  
*Humidity3pm* and *Humidity9am* take precedence, highlighting the role of **moisture levels at different times** in refining predictions.

**Step 5:**  
*WindGustSpeed* becomes critical, signaling a focus on **gust dynamics** to finalize the prediction.

**Conclusion**
The model adaptively selects features at each step, beginning with *current conditions*, transitioning to broader weather patterns, and ending with immediate indicators like *wind gusts*.

#### Approach 4: LIME

* Briefly summarize the approach.

Local surrogate models are interpretable models that work on the instance level. LIME generates predictions for a pertubed version of your dataset and then trains an interpretable model on that data that explains the predictions while trying to stay as close to them as possible. The results give an explanation into how the prediction of a single sample is affected by its feature values.

We compare those results to TabNet's feature masks, which also help with instance-wise interpretation of predictions.

* Categorize this explainability approach according to the criteria by Hohman et al.

  * WHY?
    * Interpretability & Explainability
    * Debugging & Improving Models
  * WHO?
    * Model Developers & Builders
    * Model Users
    * Non-Experts
  * WHAT?
    * Learned Model Parameters
  * HOW?
    * Instance-based Analysis & Exploration
    * Algorithms for Attribution & Feature Visualization
  * WHEN?
    * After Training
  * WHERE?
    * Tabular Data

* Interpret the results here. How does it help to explain your model?

Both the Lime feature importance and the TabNet attention masks show that they agree on the importance of most of the features for our sample (row 4242 of our data). The most important features accroding to both mehods are *Humidity3pm*, *WindGustSpeed*, *Rainfall* and *Pressure3pm*. The TabNet models masks indicate that *WindGustDir* is also important, while Lime does not show this feature as important. This could be due to the fact that Lime is a local surrogate model and might not be able to capture the importance of this feature for this specific instance.

Lime has the advantage of being able to explain for which class the features are important. For example, it shows that *Humidity3pm* is important for the prediction of No Rain tomorrow, while *Pressure3pm* is important for the prediction of Rain tomorrow. The TabNet masks do not provide this information.

In the misclassification example, the model predicted Rain tomorrow, while the true label was No Rain. The most important features for predicting Rain for this sample are *Humidity3pm*, *Rainfall*, *WindSpeed3pm* according to Lime. The TabNet masks also show that *Humidity3pm* is important, but ignore *Rainfall* and *WindSpeed3pm*. This could indicate why the model misclassified this sample.

### Summary of Approaches 

We looked at all the interpretability mechanisms TabNet has to offer and compared their results to general explanation methods we learned in the lecture. We found that the model produces reliable explanations for its predictions while lacking the detailed information that methods such as LIME provide. The explanations and predictions for the weather are in line with scientific models in that domain.

### Presentation Video Link

Provide the link to your recorded video of the presentation. This could, for instance, be a Google Drive link. Make sure it is actually sharable and test it, e.g., by trying to access it from an incognito browser tab.

[https://drive.google.com/file/d/1zNbSR19zW1jcOLplcTu9cqXLxNpvOlvi/view](https://drive.google.com/file/d/1zNbSR19zW1jcOLplcTu9cqXLxNpvOlvi/view)
