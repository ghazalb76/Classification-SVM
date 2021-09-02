# Classification-SVM
In this project, I have used <a href="https://github.com/ghazalb76/Classification-SVM/blob/main/insurance_claims.csv">*Insurance Claim Dataset*</a>, related to the transactions of an insurance company in the United States to detect fraud. Following steps show the implementation process in detail:

* Preprocessing: One-Hot Encoding and filling or removing missing data
* Seperating train data and test data
* Generating primary SVM
* Evaluating the model using **Confusion Matrix** for each of the 4 kernels: linear, poly, rbf, sigmoid
* Optimizing values of the parameters using "GridSearchCV()" method
* Regenerating the SVM based on best optimized parameters
* Drawing ROC for both primary and final SVM model


In addition, I have degraded the features' degree to 2D using PCA:
<!-- 
<p align="center">
  <img src="https://github.com/ghazalb76/DecisionTree-HeartDisease/blob/main/resultPics/Capture7.PNG">
</p> -->


By running the code using following command, you can see the results by yourself:

```
    python src.py
```

But before that you need to have the following Python packages installed:
* pandas >= 0.25.1
* numpy >= 1.17.2

A very complete **Persian Report** is also included in Report.pdf. 


Thanks to Mr. Josh Starmer for his wonderful tutorials :D
