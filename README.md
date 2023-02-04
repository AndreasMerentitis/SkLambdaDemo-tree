# Serverless Machine Learning on AWS Lambda with TensorFlow (Logistic model)

Configured to deploy a TensorFlow model to AWS Lambda using the Serverless framework.
The model is a LinearClassifier Logistic model trained in the Lambda. 

by: Mike Moritz

updates by: Andreas Merentitis (ubuntu 20.04, py37)

![relative path 1](/ml_train.jpeg?raw=true "ml_train.jpeg")
![relative path 2](/ml_predict.jpeg?raw=true "ml_predict.jpeg")


### Prerequisites

#### Setup serverless

```  
sudo npm install -g serverless

sudo serverless plugin install -n serverless-python-requirements

pip install -r requirements.txt

```
#### Setup AWS credentials

Make sure you have the AWS access key and secrete keys setup locally, following this video [here](https://www.youtube.com/watch?v=KngM5bfpttA)

### Download the code locally

```  
serverless create --template-url https://github.com/AndreasMerentitis/SkLambdaDemo-tree --path sk-lambda
```

### Update S3 bucket to unique name
In serverless.yml:
```  
  environment:
    BUCKET: <your_unique_bucket_name> 
```

### Check the file syntax for any files changed 
```
pyflakes infer.py

```
We can ignore the warning about not using 'unzip_requirements' as its needed to set the requirements for lamda 


### Deploy to the cloud  


```
cd sk-lambda

npm install

sudo serverless deploy --stage dev

curl -X POST https://t3r9pasalk.execute-api.eu-west-1.amazonaws.com/dev/upload

curl -X POST https://t3r9pasalk.execute-api.eu-west-1.amazonaws.com/dev/train

curl -X POST https://t3r9pasalk.execute-api.eu-west-1.amazonaws.com/dev/infer -d '{"epoch": "1556995767", "input": {"age": ["34"], "workclass": ["Private"], "fnlwgt": ["357145"], "education": ["Bachelors"], "education_num": ["13"], "marital_status": ["Married-civ-spouse"], "occupation": ["Prof-specialty"], "relationship": ["Wife"], "race": ["White"], "gender": ["Female"], "capital_gain": ["0"], "capital_loss": ["0"], "hours_per_week": ["50"], "native_country": ["United-States"], "income_bracket": [">50K"]}}'

python client_api_train.py

python client_api_infer.py
```

### Clean up (remove deployment and artifacts) 


```
aws s3 rm s3://serverless-ml-1 --recursive

sudo serverless remove --stage dev 
```

# Using data and extending the basic idea from these sources:
* https://github.com/mikepm35/TfLambdaDemo
* https://medium.com/@mike.p.moritz/running-tensorflow-on-aws-lambda-using-serverless-5acf20e00033
* https://aws.amazon.com/blogs/machine-learning/how-to-deploy-deep-learning-models-with-aws-lambda-and-tensorflow/









