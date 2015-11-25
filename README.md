# Twitter sentiment analysis using *[weka](http://www.cs.waikato.ac.nz/ml/weka/)* API
## Overview
I will try to give a short introduction to how I constructed a SVM with weka + libsvm to do sentiment analysis on tweets.

To do sentiment analysis on tweets, that is, predict the sentiment of any tweet if it is either positive or negative, we have to train a classifier with a set of handlabeled tweets. Each tweet has been labeled with a class attribute of '-1' for negative sentiment or '1' for positive sentiment. 

The training set is then passed through a cleansing process, eliminating all nonsense characters and words (like @mentions and URLs) from the text. This is done using *regular expressions*. 
In the next phase the text is being tokenized and converted into a [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) vector. All these tf-idf vectors, together with their label, are then used to train a support vector machine (SVM) with [radial basis function kernel (RBF)](https://en.wikipedia.org/wiki/Radial_basis_function_kernel).

One can use then the trained SVM to classify new tweets.

## The ML-Framework
We'll be using [weka](http://www.cs.waikato.ac.nz/ml/weka/) for all the hard stuff. To add weka to your java project edit the maven `pom.xml` and add
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

Here are some links to the API documentation of the weka project:
* [project homepage](http://www.cs.waikato.ac.nz/ml/weka/)
* [weka FAQ](https://weka.wikispaces.com/Frequently+Asked+Questions)
* [using weka in your Java code](https://weka.wikispaces.com/Use+WEKA+in+your+Java+code)
* [programmatic use](https://weka.wikispaces.com/Programmatic+Use)
* [weka Javadoc](http://weka.sourceforge.net/doc.stable/)
* [youtube video tutorials on weka by Rushdi Shams](https://www.youtube.com/playlist?list=PLJbE6j2EG1pZnBhOg3_Rb63WLCprtyJag)

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
		
		...
	}
	
	
}
```

