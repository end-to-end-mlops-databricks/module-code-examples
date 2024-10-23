# Week 2 Materials and Code Overview

Hello students,

Before diving into the Week 2 materials and code, let's clarify the key concepts and implementations covered in this lecture.

## Overview

We explored three different use cases (implementations) that demonstrate basic and complex scenarios for registering models with MLflow. For Databricks model deployment, we use MLflow for experiment tracking and model registry, which are essential components as mentioned in the first lecture.

## Code Structure and Implementations

### 1. Data Preparation
```
02.prepare_dataset.py
```
We extended our preprocessing code from the previous lecture, adding functionality to save test and train set tables in Unity Catalog.

### 2. Basic MLflow Experiment
```
02.mlflow_experiment.py
```
This script demonstrates a simple MLflow experiment. When executed, it creates an experiment with information for each run, visible in the Databricks Experiments section. Start with this to familiarize yourself with MLflow basics.


### 3. Logging and Registering a Model
```
03.log_and_register_model.py
```
Steps:
- We start by loading our train and test datasets, then we train an sklearn pipeline within an MLflow run. During this run, we set 2 tags. Tags are a useful feature in MLflow for attaching additional information to your run. For example, we tag the Git commit and branch name to improve traceability.
- Logging Metrics and Data: After the run is complete, we log the desired metrics. These metrics are visible in the UI, in Databricks experiments. Additionally, we log the dataset we used for this run to keep track of the data.
- Model Registration: Next, we show how to register a model. MLflow creates model artifacts during the run, and by using the model_uri from that specific run, we register the model and obtain its version.
- Retrieving Data: Finally, we show how to retrieve the data from the MLflow run, as it was logged during the previous step.

### 4. Logging and Registering a Model
```
04.log_and_register_custom_model.py
```
This is a more advanced use case. In this implementation, we explain two additional aspects.

**Registering a Custom Model**: While MLflow provides built-in flavors for commonly used ML libraries, if you're using your own custom code for training and registering a model, you need to use the pyfunc flavor. We show how to wrap a model for this purpose. In our example, we trained an sklearn model (which isn't custom), but we treat it as if it were a custom model. We assume the trained model is custom and register it using pyfunc. Another reason for using this method could be that you want to control how predictions are returned, rather than using the default output from the library. For instance, we demonstrate returning predictions as a dictionary, {predictions: [a, b, c, d]}, instead of a simple list, [a, b, c, d]. The wrapper class allows this customization.

**Using a Custom Library**: The second thing we show is how to use a custom Python library in your MLflow run. Suppose you have a custom function for transforming predictions or preprocessing inputs before calling _model.predict_. If this custom functionality is part of a Python module you've developed, you can make it available in your MLflow run by passing it as a pip requirement. In the pyfunc wrapper class, you can then import and use your custom module like any other Python library. In our example, we wanted to use _adjust_predictions_ function after predictions are generated. We packaged our house_price module, which contains this function, and provided as a pip requirement. The path we pass, is the location of .whl file of our package in Volumes.

### 5. Logging and Registering with feature look up
```05.log_and_register_fe_model.py```
This script shows how to log and register a model with feature look up. Feaure look up is used when you need certain features at inference time. This is often a situation for real-time models, rather than batch models. 

Let’s consider the case where you want to predict ice cream demand. While you might have certain real-time data at the moment of prediction, such as:
Current temperature, Time of day, Location.

You might also need additional features that aren’t part of your real-time input data but are critical for accurate predictions. These features could include:

Historical Sales Data: You might need the sales of ice cream for the same day and location over the last 5 years.

Weather Forecast: Even though you have the current temperature, you might also want to retrieve weather forecasts for the coming days, as it could influence ice cream demand.

Holiday Information: If there’s a holiday in the coming days, the demand for ice cream may spike, and this data needs to be fetched dynamically at inference.

In these situations, the model would not have all the necessary data in real-time, and this is where feature lookup becomes important. The model can fetch the missing features from a feature store based on the input data (like location or date) and use them during inference.

In our example, we are using a house price dataset, which consists of static features. Since the features in this dataset don’t change over time (e.g., the number of rooms, house size, or location), it wouldn’t naturally fit into a real-time feature lookup scenario. However, we still mimic the feature lookup implementation to explain how it would work in practice.

This way, you can still learn the process of registering and logging models with feature lookup, which is useful for more dynamic scenarios, like demand forecasting or real-time recommendation systems, where features need to be fetched at inference time.