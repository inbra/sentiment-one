# Twitter sentiment analysis with *[weka](http://www.cs.waikato.ac.nz/ml/weka/)* API
## Overview
I will try to give a short introduction to how I constructed a SVM with weka + libsvm to do sentiment analysis on tweets.

To do sentiment analysis on tweets, that is, predict the sentiment of any tweet if it is either positive or negative, we have to train a classifier with a set of handlabeled tweets. Each tweet has been labeled with a class attribute of '-1' for negative sentiment or '1' for positive sentiment. 

The training set is then passed through a cleansing process, eliminating all nonsense characters and words (like @mentions and URLs) from the text. This is done using *regular expressions*. 
In the next phase the text is being tokenized and converted into a [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) vector. All these tf-idf vectors, together with their label, are then used to train a support vector machine (SVM) with [radial basis function kernel (RBF)](https://en.wikipedia.org/wiki/Radial_basis_function_kernel).

One can use then the trained SVM to classify new tweets.

## The ML-Framework
We'll be using [weka](http://www.cs.waikato.ac.nz/ml/weka/) for all the hard stuff. To add weka to your java project edit the maven pom.xml and add
```xml
		<dependency>
			<groupId>nz.ac.waikato.cms.weka</groupId>
			<artifactId>weka-stable</artifactId>
			<version>3.6.13</version>
		</dependency>
```
Weka comes with a wrapper for [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) - which we will use for contructing the support vector machine. Add libsvm to your project using...
```xml
		<dependency>
		  <groupId>com.facebook.thirdparty</groupId>
		  <artifactId>libsvm</artifactId>
		  <version>3.18.1</version>
		</dependency>
```

