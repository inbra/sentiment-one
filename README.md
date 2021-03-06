# Twitter sentiment analysis using *[weka](http://www.cs.waikato.ac.nz/ml/weka/)* API
<a name="toc"></a>
## Contents
* [Table of contents](#toc)
* [Overview](#overview)
* [Useful links](#linklist)
* [Basics](#basics)
* [Text cleansing](#cleansing)
* [Set up and train the classifier](#train)



<a name="overview"></a>
## Overview
I will try to give a short introduction to how I constructed a SVM with weka + libsvm to do sentiment analysis on tweets.

To do sentiment analysis on tweets, that is, predict the sentiment of any tweet if it is either positive or negative, we have to train a classifier with a set of handlabeled tweets. Each tweet has been labeled with a class attribute of '-1' for negative sentiment or '1' for positive sentiment. 

The training set is then passed through a cleansing process, eliminating all nonsense characters and words (like @mentions and URLs) from the text. This is done using *regular expressions*. 
In the next phase the text is being tokenized and converted into a [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) vector. All these tf-idf vectors, together with their label, are then used to train a support vector machine (SVM) with [radial basis function kernel (RBF)](https://en.wikipedia.org/wiki/Radial_basis_function_kernel).

One can use then the trained SVM to classify new tweets.


<a name="ml-framework"></a>
## The ML-Framework
We'll be using [weka](http://www.cs.waikato.ac.nz/ml/weka/) for all the hard stuff. To add weka to your java project edit the maven `pom.xml` and add
```xml
		<dependency>
			<groupId>nz.ac.waikato.cms.weka</groupId>
			<artifactId>weka-stable</artifactId>
			<version>3.6.13</version>
		</dependency>
```
Weka comes with a wrapper for [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) - which we will use for constructing the support vector machine. Add libsvm to your project using...
```xml
		<dependency>
		  <groupId>com.facebook.thirdparty</groupId>
		  <artifactId>libsvm</artifactId>
		  <version>3.18.1</version>
		</dependency>
```


<a name="linklist"></a>
## Useful links
Here are some links to the API documentation of the weka project:
* [project homepage](http://www.cs.waikato.ac.nz/ml/weka/)
* [weka FAQ](https://weka.wikispaces.com/Frequently+Asked+Questions)
* [using weka in your Java code](https://weka.wikispaces.com/Use+WEKA+in+your+Java+code)
* [programmatic use](https://weka.wikispaces.com/Programmatic+Use)
* [writing your own filter](https://weka.wikispaces.com/Writing+your+own+Filter)
* [weka Javadoc](http://weka.sourceforge.net/doc.stable/)
* [youtube video tutorials on weka by Rushdi Shams](https://www.youtube.com/playlist?list=PLJbE6j2EG1pZnBhOg3_Rb63WLCprtyJag)


<a name="basics"></a>
## Basics
### ARFF files
Reading data from [ARFF](https://weka.wikispaces.com/ARFF) files is straightforward: 
```java
	Instances data = null;
	String filename = "datafile.arff";
	ArffLoader loader = new ArffLoader();

	loader.setFile(new File(filename));
	data = loader.getDataSet();
```
See also the documentation [here](https://weka.wikispaces.com/Use+Weka+in+your+Java+code#Instances-ARFF File).


### Data `Instances`
> `Class for handling an ordered set of weighted instances.`

An object of type `Instances` is the representation of an ARFF file once loaded- it holds the relation (table-) name, a list of attributes and all data instances.

<a name="instance"></a>
### create an `Instance` for classifying
`Instance` represents one "row" of the relation. To classify an instance with the trained classifier, the new instance has to be given the same schema as the data set used during training.
```java
	// create new text attribute (column)
	Attribute aText = new Attribute("text",(FastVector) null);

	// create new class attribute (to store the classifier's result):
	FastVector fvClasses = new FastVector(2);
	fvClasses.addElement("-1");
	fvClasses.addElement("1");
	Attribute aSent = new Attribute("sentiment",fvClasses);

	// set up schema:
	FastVector fvAtt = new FastVector(2);
	fvAtt.addElement(aText);
	fvAtt.addElement(aSent);
	
	// create "test-set" of size 1
	Instances testSet = new Instances("prediction", fvAtt, 1);
	testSet.setClassIndex(testSet.numAttributes()-1);

	// add the only instance
	Instance instance = new SparseInstance(testSet.numAttributes());
	instance.setValue((Attribute) fvAtt.elementAt(0), "Text to be classified");
	testSet.add(instance);
```


### access String attribute of an `Instance`
```java
	int attributeIndex = 0;
	
	// get String value from instance
	String text = instance.stringValue(attributeIndex);
	
	String modifiedText = doSomething(text);
	
	// set String value
	instance.setValue(attributeIndex, modifiedText);
```


<a name="cleansing"></a>
## Text cleansing
First we have to get rid of some noise in the tweet's text. We can apply custom filters (regular expression substitution) by modifying the `input()` method of the `StringToWordVector` class, that is used to tokenize the text and create the TF-IDF matrix. 

```java
public class MyStringToWordVector extends StringToWordVector {
	
	@Override
	public boolean input(Instance instance) throws Exception {
	
		String text = instance.stringValue(0);
		text = normalize(text).trim();
		instance.setValue(0, text);
		
		return super.input(instance);
	}
	
	private String normalizeText (String text){
	
		text = text.toLowerCase();				// only lowercase chars allowed
		text = Normalizer.normalize(text, Normalizer.Form.NFD); // translate UTF-8 chars properly
		
		// get rid of '@mentions'
		Pattern p = Pattern.compile("@\\w+ *");
		Matcher m = p.matcher(text);
		text = m.replaceAll("");
		
		// eliminate URLs
		Pattern p = Pattern.compile("http[^ ]*");
		Matcher m = p.matcher(text);
		text = m.replaceAll("");
		
		// do some more cleaning ...
		// [...]
		
		return text;
	}
	
	
}
```

One would typically use the `MyStringToWordVector` -Filter like this:
```java
	Instances data = 				// get data from somewhere
	NGramTokenizer tokenizer = createTokenizer();	// get a tokenizer
	SnowballStemmer stemmer = createStemmer();	// get a word-stemmer
	
	MyStringToWordVector filter = new MyStringToWordVector();
	
	filter.setTokenizer(tokenizer);
	filter.setInputFormat(data); 		// pass the schema of the data to the filter; throws exception
	filter.setWordsToKeep(1000000);		// recommended for production: 1000000
	filter.setDoNotOperateOnPerClassBasis(true);
	filter.setLowerCaseTokens(true);
	filter.setAttributeIndicesArray(new int[] {0}); // tell the filter on which column(s) to work
	filter.setIDFTransform(true);		// we want IDF data, so: true
	filter.setTFTransform(false);		// false, since we want pure TF (term frequency)
	filter.setOutputWordCounts(true);
	filter.setStopwords(new File(this.pwd+this.stopwordFile));
```
And then either pass the filter to a `FilteredClassifier` as I'll explain later or apply it directly on the data instances, like so:<br>`data = Filter.useFilter(data, filter);`


<a name="train"></a>
## Set up and train the classifier
In order to preprocess and classify some tweet text, the classifier has to be trained first. We will use the meta classifier `FilteredClassifier` from the weka framework since it stores not only the trained model but _also the filter with the learned feature space_.

First create a class derived from `FilteredClassifier` to get access to the feature space (filtered instances) and be able to save it to a file:
```java
	public class MyFilteredClassifier extends FilteredClassifier {
		public Instances getFilteredInstances(){
			return m_FilteredInstances;
		}
	}
```

Set up the training:
```java
	// configure SVM
	LibSVM svm = new LibSVM();
	svm.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_RBF,LibSVM.TAGS_KERNELTYPE)); // type: RBF
	svm.setCost(100.0);		// the C parameter
	svm.setProbabilityEstimates(false);
	svm.setDoNotReplaceMissingValues(true);
	
	// set up FilteredClassifier
	FilteredClassifier classifier = new MyFilteredClassifier();
	classifier.setFilter(filter);	// the MyStringToWordVector filter 
	classifier.setClassifier(svm);  // pass the svm to the meta construct
```

Training is done in weka using the `Evaluation` class, which can be used as well to print some statistics about the training:
```java
	Instances sample = 	// the training set from somewhere
	int numFolds = 5; 	// 5 folds make a 80/20 separation between train- and test-set
	
	// setting up train- and test-set
	Random rand = new Random(seed);
	sample.randomize(rand);
	sample.stratify(numFolds);

	Instances trainingSet 	= sample.trainCV(numFolds, 0);
	Instances testSet	= sample.testCV(numFolds, 0);

	classifier.buildClassifier(trainingSet);

	// create new Evaluation object and pass the schema of the dataset
	Evaluation eval = new Evaluation(trainingSet);
	
	// evaluate classifier on test-set
	eval.evaluateModel(classifier, testSet);
	
	// print some stats about the result:
	System.out.println(eval.toSummaryString());
	// more details:
	System.out.println(eval.toClassDetailsString());
	// print confusion matrix
	System.out.println(eval.toMatrixString());
```

Once the classifier is trained it can be saved to a file.
```java
	weka.core.SerializationHelper.write("classifier.model", classifier);
```

Save the trained dictionary (feature space, attribute list) as well:
```java
	Instances attributeList = ((MyFilteredClassifier) classifier).getFilteredInstances();
	PrintWriter writer = new PrintWriter(new FileWriter("attributeList.arff"));
	writer.print(attributeList);
	writer.close();
```


<a name="classify"></a>
## Use the classifier
I'll assume the trained classifier has been saved to a file and will be recovered like so:
```java
	MyFilteredClassifier classifier = 
		(MyFilteredClassifier) weka.core.SerializationHelper.read("classifier.model");
```

Before being able to classify a String (passed as a parameter, read from file) it has to be [turned into an `Instance`](#instance). The newly created one-instance test set can than be classified easily.
```java
	Evaluation eval = new Evaluation(((MyFilteredClassifier) classifier).getFilteredInstances()); // set schema
	probs = classifier.distributionForInstance(oneInstancetestSet.firstInstance());
```
