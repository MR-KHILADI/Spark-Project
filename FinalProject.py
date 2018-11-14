
# coding: utf-8

# ## Data Collection, Creating a Word Cloud, Sentiment Analysis and Document Clustering using Naive Bayes Logistic Regression and Random Forest
# ### 1. Vatsal Raval (vraval)
# ### 2. Amit Pathak (amitpath)
# ### 3. Akshay Agrawal (aa267)
# ### 4. Prem Shah (prembhar)

from nytimesarticle import articleAPI
api = articleAPI('773cf44039f544a6bad56fd1f4c503fe')


# Scraper function

def scraper(url):
    r  = requests.get(url)

    data = r.text
    soup = BeautifulSoup(data)
    output = ""
    for link in soup.find_all('p', attrs={'class': 'css-1cy1v93 e2kc3sl0'}):  # extracting paragraphs with class css-1cy1v93 e2kc3sl0
        output += (link.text)
    return output

j = 0

articles = api.search( q = 'Sexual Abuse',begin_date = 20170601,page=0)

for i in articles['response']['docs']:
    if i['web_url'] != "":
        filename = "test" + str(j) + ".csv"  # this will save every article in different csv files
        f = open(filename, 'a+')
        text = scraper(i['web_url'])
        f.write(text.encode("utf-8"))
        f.write('\n')
        f.close()
    j = j +1

import csv
from textblob import TextBlob

with open(r"C:\Users\iamva\Jupyter Notebook\test.csv","r", encoding='utf8') as f:
    cr = csv.reader(f,delimiter=",") # , is default

    for row in rows:
        text = row[0]
        blob = TextBlob(text.encode("utf-8"))
        print (text.encode("utf-8"))
        print (blob.sentiment.polarity, blob.sentiment.subjectivity)

data = pd.read_csv('test.csv')
# Keeping only the neccessary columns
data = data[['text','sentiment']]

# Splitting the dataset into train and test set
train, test = train_test_split(data,test_size = 0.1)
# Removing neutral sentiments
train = train[train.sentiment != "Neutral"]

####### For getting a Wordcount ##################

from mrjob.job import MRJob
import re
x=[ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
   "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
   "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", 
   "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here",
   "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", 
   "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", 
   "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", 
   "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", 
   "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", 
   "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", 
   "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", 
   "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", 
   "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ];
WORD_REGEXP = re.compile(r"[a-zA-Z]+")

class MRWordFrequencyCount(MRJob):

    def mapper(self, _, line):
        words = WORD_REGEXP.findall(line)
        for word in words:
            word=word.lower()
            if (word not in x):
                if len(word) >4:
                    if len(word)<15:
                        yield word, 1

    def reducer(self, key, values):
        j = sum(values)
        if (j >5):
            yield key,j
          
import pandas as pd
data=pd.read_table('output4.txt',header=None,skipinitialspace=True)
data = data.rename(columns={1: 'count'})
print
x=data.nlargest(50,'count')
print(x)
x.to_csv('top50_4.txt', sep='\t', encoding='utf-8')




neg_cnt = 0
pos_cnt = 0
for obj in test_neg: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Negative'): 
        neg_cnt = neg_cnt + 1
for obj in test_pos: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Positive'): 
        pos_cnt = pos_cnt + 1
        
print('[Negative]: %s/%s '  % (len(test_neg),neg_cnt))        
print('[Positive]: %s/%s '  % (len(test_pos),pos_cnt))


# Extension to the project : Document Clustering using Spark API's in Python

from pyspark.sql import SQLContext
from pyspark import SparkContext
sc =SparkContext()
sqlContext = SQLContext(sc)

#### loading data 
data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('Data.csv')
data.show(5)
data.printSchema()

## using sql query counting the number of articles based which is grouped by category
from pyspark.sql.functions import col

data.groupBy("category") \
    .count() \
    .orderBy(col("count").desc()) \
    .show()
    
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression

# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="article", outputCol="words", pattern="\\W")

# stop words
add_stopwords = ["http","https","amp","rt","t","c","the"] 

stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(add_stopwords)



from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF

label_stringIdx = StringIndexer(inputCol = "category", outputCol = "label")

#### using TF IDF


hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
pipeline = Pipeline(stages=[regexTokenizer, stopwordsRemover, hashingTF, idf, label_stringIdx])

pipelineFit = pipeline.fit(data)
dataset = pipelineFit.transform(data)
dataset.show(5)

(trainingData, testData) = dataset.randomSplit([0.8, 0.2], seed = 123)
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))


print('now at this point we have we split our data as 80% training and 20% test data') 
print('At this point we are ready to build logistic regression, naive bayes and random forest model and check prediction of test set')
  
  
### using Logistic Regression  
lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainingData)

predictions = lrModel.transform(testData)

predictions.filter(predictions['prediction'] == 0) \
    .select("article","Category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)



evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
result2 = evaluator.evaluate(predictions)
print ('Logistic Regression model gives accuracy = '+str(result2))

## using naivebayes
from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes(smoothing=1)
model = nb.fit(trainingData)

predictions = model.transform(testData)
predictions.filter(predictions['prediction'] == 0) \
    .select("article","Category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)



evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
result3 = evaluator.evaluate(predictions)
print ('naive bayes  accuracy is: '+str(result3))

## random forest
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="label", \
                            featuresCol="features", \
                            numTrees = 100, \
                            maxDepth = 4, \
                            maxBins = 32)

# Train model with Training Data
rfModel = rf.fit(trainingData)

predictions = rfModel.transform(testData)

predictions.filter(predictions['prediction'] == 0) \
    .select("article","Category","probability","label","prediction") \
    .orderBy("probability", ascending=False) \
    .show(n = 10, truncate = 30)
    
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
result4 = evaluator.evaluate(predictions)
print ('random forest accuracy is: '+str(result4))


