# Amazon SageMaker Pipelines: Deploying End-to-End Machine Learning Pipelines in the Cloud


# Introduction

Cloud computing is one of the fastest growing skills in the Machine Learning world. Among cloud services companies, Amazon stands out for providing one of the most advanced tools for Machine Learning: Amazon SageMaker. Using SageMaker you can, among many other things, build, test and deploy Machine Learning models. Furthermore, you can create End-to-End pipelines in order to integrate your models in a CI/CD process.

In this repository we are going to use Amazon SageMaker to create an End-to-End pipeline step by step. First, we will look into an overview of the project, then we will go with some theoretical explanations, and last but not least, we will code. For the coding section we will use [this Jupyter notebook](https://github.com/Adricarpin/SageMaker-Pipelines/blob/master/SageMaker-Pipeline.ipynb).


We will work with [Adult Census Income](https://www.kaggle.com/uciml/adult-census-income) Dataset. We will use 'income', a binary variable that explains if a person earns more than 50k or not, as the target variable. For the training step, we will use an image of XGBoost provided by AWS. 





# Overview of the project


Before explaining how to build a pipeline, I think is crucial to first understand the plan. A little further down you will find a diagram of the pipeline that is going to be our blueprint for creating one. If you can understand that diagram, half of the job is done (the rest is just put it into code :see_no_evil: ). But, before looking at the diagram, I think it would be benefitial to first understand these things:

- Which are the steps in our pipeline
- Which are the inputs and outputs for each step

To understand this, we are going to explain in an intuitive way all steps in the pipeline:

:ledger:NOTE: Try to remember the file names as it would be easier to you to then understand the diagram.


1. **The Preprocessing Step**: In this step we will preprocess the raw data. Therefore, the input in this step is the raw data, and the output is the processed data (data ready for passing it to the model). In our project, the raw data is the "adult.csv" file, and the processed data files names will be "train.csv", "validation.csv" and "test.csv".


2. **The Training Step**: In this step we will train the model. The input in this step is the processed data, in particular, the "train.csv" and "validation.csv" files. The output will be our trained model. This file is called "model.tar.gz".


3. **The Evaluation Step**: In this step we will test our model with new data. The input will be the model file and the file with test data, that is, "model.tar.gz" and "test.csv". The output will be a file where the metadata of this step (for example the accuracy) is stored. we will call it "evaluation.json".

4. **The Condition Step**: We have to know if our model is good enough before delivering it. In the Condition Step we compare the test accuracy with a threshold. If the accuracy is higher than the threshold, we continue with the next step. If it isn't, we stop the pipeline so the model is not deployed.

5. **The Create Model Step**: If the model passes the Condition Step, we create a SageMaker Model. A SageMaker Model is an instance that can be deployed to an Endpoint. 

6. **The Register Model Step**: If the model passes the Condition Step, we will register the model so we can access it whenever. In SageMaker model registry you can have a catalog of models with their corresponding metadata.

Once we know all of this, let's see the  long-awaited pipeline diagram:

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/135580574-5ab27c4e-ffbd-469e-aaba-0c839f57ea6e.png" />
</p>


:ledger:NOTE: "Create Model Step" and "Register Model Step" are independent steps: there is no specific order between them. 

Ok, now that we are aware of the plan, let's go for it!



# SageMaker Pipelines Theory: What we need to know

This section simply tries to anwer the following question: What you have to know about SageMaker pipelines before start coding? 


First you have to understand that the way SageMaker build pipelines is by specifying the steps of the pipeline first and then connecting them with a [pipeline instance](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-pipeline.html).

In order to create pipeline steps we use [steps classes](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html). There are two step classes that are particularly important to discuss before starting to code: the ```ProcessingStep``` class and the ```TrainingStep``` class.



## The ProcessingStep class

For our "Preprocessing Step" and "Evaluation Step" we will use a ```ProcessingStep``` class (you can see the documentation [here](https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html)). Basically, This is the class that SageMaker uses to build the steps where we process  data. When building a ```ProcessingStep```, you have to pass a Processor instance and the code. The code is just a Python script where you process the data. A Processor instance is a Docker image with the specifications to run the step. 

Wait, so I have to create a Docker image? 

You can create one if you want, but you don't need to. You can just use a Processor instance provided and maintained by SageMaker. The most commonly used are [SKLearnProcessor](https://docs.aws.amazon.com/sagemaker/latest/dg/use-scikit-learn-processing-container.html) (for SKLearn) and [PySparkProcessor](https://docs.aws.amazon.com/sagemaker/latest/dg/use-spark-processing-container.html) (for Apache Spark). For importing a specific image, you can use the [ScriptProcessor](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-container-run-scripts.html) instance.


:ledger:NOTE: In our project, we will pass a ```SKLearnProcessor``` instance for the preprocessing step, and a ```ScriptProcessor``` for the evaluation step. For the ```ScriptProcessor``` we will pass a XGBoost image provided by SageMaker.


For more information about this topic you can go [here](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html). If something is missing, don't worry, Once you go through the hands-on section everything will become clearer.



## The TrainingStep class


For the training step we will use the ```TrainingStep``` class (you can see the documentation [here](https://aws-step-functions-data-science-sdk.readthedocs.io/en/stable/sagemaker.html)). When specifying a ```TrainingStep``` class, you have to pass an estimator.

There are 3 ways to build an estimator:

1. **Use Build-in algorithms**: you can import an image of an estimator from the SageMaker repository. 

2. **Use script mode in a supported framework**: you can build your own algorithm in a Python script, and use a framework that supports it. This allows more flexibility than the build-in algorithms.


3. **Bring your own container**: Finally, you can create your own container. This option brings more flexibility than the script mode and the build-in algorithm options.


As in our project we are going to create a XGBoost estimator, you can check the different ways to create it [here](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html). You can also check [SageMaker training documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html) for more information.



Ok, now that we have all this information in our brain, time to code!


# Hands On with SageMaker

:warning:**WARNING**:warning: : If you are reading this and you are a total beginner with AWS, **be careful**: you are going to use a paid service so you should know how to manage it. If you know the basics of AWS (S3, EC2, IAM roles, Billing...) this exercise shouldn't be a problem. You can do it under the SageMaker free tier without paying any money. **But**, be careful and **remove everything once you finish**, because if you don't do it sooner or later AWS will start charging you. I highly recommend you to set a **billing alarm** that sends you an email once the charges are above a specific threshold, just in case. If you are a total beginner with AWS and you want to learn the fundamentals, I highly recommend you [this course](https://www.coursera.org/learn/aws-cloud-technical-essentials?specialization=aws-fundamentals).


If you are confident and ready, let's go for it! 


First thing we have to do is to create a SageMaker Studio (if you already know how to do it you can skip this part). In the AWS search bar, search for SageMaker and click on Amazon SageMaker. Once you are in, click on Amazon SageMaker Studio on the sidebar:

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/135505558-8707bd23-fd6f-49ec-a913-757ea05acbe2.png" />
</p>



You can create a Studio with the Quick start option just entering a name and choosing an IAM role with an AmazonSageMakerFullAccess policy attached.


:ledger:NOTE: If you don't know how to create a IAM role, you should before moving on, so be careful. IAM roles is a tool that you need to understand as it can handle security problems. Learning how IAM roles work can be a chore, but remember that AWS is a very powerfull tool. With great power comes great responsability. Don't forget uncle Ben.


Once the Studio is created, click on Open studio.

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/135506594-2a8a4a33-d068-424e-bca9-f5b750d4e208.png" />
</p>


Now that you are in, you can start a Machine Learning project!


For our project you will have to import a Jupyter notebook and the raw data. You can just clone this [Github repository](https://github.com/Adricarpin/SageMaker-Pipelines) into Amazon Sagemaker Studio by going to Git, Clone a repository,

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/135500788-3eb1ce69-124f-4b0d-b50b-31698d196469.png" />
</p>



and paste the URL of the repository. Once you click "Clone" you should see a folder in the sidebar with the notebook and the data.


Last but not least, you can choose the Kernel by going to Kernel, Change Kernel...

<p align="center">
  <img src="https://user-images.githubusercontent.com/86348959/135503204-102ec88e-1b3d-4bb0-a18c-c57412375186.png" />
</p>




Python 3 (Data Science) works for me.

Now you are ready to run the notebook and build a SageMaker Pipeline! 

I now, you have a complete notebook in your hands and maybe you will want to run every cell in one go and look if it works, but remember what the best practices are. I highly encourage you to read slowly the code and comments and extract the gist of it. Then play with the code: I challenge you to try to replicate the pipeline with a different dataset, or to change the Build-in estimator to an estimator made with the "script mode in a supported framework" option (see [Use XGBoost as a framework](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)). You can also try to implement [new steps](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-tuning).

Once you end with the notebook, you will know how to create a pipeline with Amazon SageMaker!


AWS is a huge world and you can always learn new things. If you want to learn more, I recommend you to check the [SageMaker developer guide](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)(specially the [pipeline section](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html)), the [Amazon SageMaker Example Notebooks](https://sagemaker-examples.readthedocs.io/en/latest/) and its [Github repository](https://github.com/aws/amazon-sagemaker-examples). You can also consider doing the [Practical Data Science ](https://www.coursera.org/specializations/practical-data-science?) specialization from DeepLearning.AI and AWS.


I hope you have learned a lot! Thanks for reading!



# References

- [Amazon SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
- [Practical Data Science Specialization](https://www.coursera.org/specializations/practical-data-science?)
- [Amazon Sagemaker Examples](https://sagemaker-examples.readthedocs.io/en/latest/)
