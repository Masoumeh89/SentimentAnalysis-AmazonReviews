# SentimentAnalysis-AmazonReviews
Sentiment Analysis of Amazon Customer Reviews Using NLTK and VADER



## Table of Contents

1. [Introduction](#introduction)
   
2. [Data Source](#data-source)
  
3. [Data Loading and NLTK Basics](#Data-loading-and-nltk-basics)
 
4. [Exploratory Data Analysis](#exploratory-data-analysis)
   
     4.1. [Count of Reviews by Stars](#count-of-reviews-by-stars)
   
     4.2. [Helpfulness Ratio Analysis](#helpfulness-ratio-analysis)
   
     4.3. [Review Count by Time](#review-count-by-time)
   
     4.4. [Distribution of Word Count in Reviews](#distribution-of-word-count-in-reviews)
   
5. [Basic NLTK Operations](#basic-nltk-operations)
    
     5.1. [Tokenization](#tokenization)
    
     5.2. [Part-of-Speech Tagging](#part-of-speech-tagging)
   
     5.3. [Named Entity Recognition](#named-entity-recognition)
   
6. [Sentiment Analysis Techniques](#sentiment-analysis-techniques)
    
     6.1. [VADER Sentiment Scoring](#vander-sentiment-scoring)
    
     6.2. [Roberta Pretrained Model](#roberta-pretrained-model)
   
      6.2.1. [Combine and compare](#combine-and-compare)
        
      6.2.2. [Review Examples](#review-examples)
        
     6.3. [The Transformers Pipeline](#the-transformers-pipeline)

7. [Conclusion](#Conclusion)




## Introduction

This project involves sentiment analysis on Amazon customer reviews using Natural Language Processing (NLP) techniques. The goal is to gain insights into the sentiment behind the reviews by analyzing their text content, performing sentiment scoring, and visualizing the results. This project utilizes the Natural Language Toolkit (NLTK) for basic NLP operations, as well as VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis.


## Data Source 

This dataset contains reviews of fine foods from Amazon, covering a period of over 10 years and including approximately 500,000 reviews up to October 2012. The reviews feature product and user details, ratings, and the text of the review itself. It also includes reviews from various other Amazon categories.

Data Link:

https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews


## Data Loading and NLTK Basics

**Import Libraries**

To begin, we import the necessary libraries for data manipulation, sentiment analysis, and visualization. These include popular libraries such as pandas, numpy, matplotlib, seaborn, and nltk.


```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

plt.style.use('ggplot')

```

**Load the Dataset**

We load a dataset of Amazon customer reviews, containing detailed information about customer experiences. For demonstration purposes, we use the first 500 reviews.

```python

df = pd.read_csv('/kaggle/input/amazon-fine-food-reviews/Reviews.csv')
df = df.head(500)
df.head()

```

## Exploratory Data Analysis

### Count of Reviews by Stars

We begin our exploratory data analysis by visualizing the count of reviews for each star rating in the dataset.

```python

ax = df['Score'].value_counts().sort_index().plot(kind='bar', color='#3CB371', title='Count of Reviews by Stars', figsize=(10, 5))
ax.set_xlabel('Review Stars')
ax.set_ylabel('Number of Reviews')
plt.show()

```
<img width="538" alt="image" src="https://github.com/user-attachments/assets/99deaf66-c32c-43f6-bf49-e98dfb1e7662">


### Helpfulness Ratio Analysis

Next, we calculate the helpfulness ratio, defined as the ratio of the helpfulness numerator to the helpfulness denominator, and analyze its distribution.

```python

df['HelpfulnessRatio'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'].replace(0, 1)
plt.figure(figsize=(10, 5))
plt.hist(df['HelpfulnessRatio'], bins=50, color='#7B68EE')
plt.title('Distribution of Helpfulness Ratio')
plt.xlabel('Helpfulness Ratio')
plt.ylabel('Number of Reviews')
plt.show()

```

<img width="563" alt="image" src="https://github.com/user-attachments/assets/79038e53-dc44-44d1-aa28-f27d3ce14b67">



### Review Count by Time

We convert the review time from UNIX timestamp to a datetime format and analyze the number of reviews over time.

```python

df['ReviewTime'] = pd.to_datetime(df['Time'], unit='s')
df.set_index('ReviewTime')['Id'].resample('M').count().plot(figsize=(10, 5), color='teal')
plt.title('Number of Reviews Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Reviews')
plt.show()


```

<img width="540" alt="image" src="https://github.com/user-attachments/assets/7b96adfd-91d1-4a2f-9670-e33c7cd342f9">




### Distribution of Word Count in Reviews

We also analyze the distribution of the number of words in the review text to understand the variability in review length.

```python

df['WordCount'] = df['Text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 5))
plt.hist(df['WordCount'], bins=50, color='skyblue')
plt.title('Distribution of Word Count in Reviews')
plt.xlabel('Word Count')
plt.ylabel('Number of Reviews')
plt.show()

```

<img width="548" alt="image" src="https://github.com/user-attachments/assets/fd48e65f-abb7-4a20-a72e-1fc5c9a3b524">


## Basic NLTK Operations


### Tokenization


We tokenize the text in the reviews, breaking it down into individual words.

```python

import nltk
from nltk.tokenize import word_tokenize

example = df['Text'][50]
print(example)

```

This oatmeal is not good. Its mushy, soft, I don't like it. Quaker Oats is the way to go.

```python

example = df['Text'][50]
tokens = nltk.word_tokenize(example)
tokens[:10]

```
['This', 'oatmeal', 'is', 'not', 'good', '.', 'Its', 'mushy', ',', 'soft']

### Part-of-Speech Tagging

We then tag each token with its part of speech (POS) to understand the grammatical structure of the text.

```python

tagged = nltk.pos_tag(tokens)
tagged[:10]

```
[('This', 'DT'),
 ('oatmeal', 'NN'),
 ('is', 'VBZ'),
 ('not', 'RB'),
 ('good', 'JJ'),
 ('.', '.'),
 ('Its', 'PRP$'),
 ('mushy', 'NN'),
 (',', ','),
 ('soft', 'JJ')]

### Named Entity Recognition

Finally, we perform named entity recognition to identify entities such as people, organizations, and locations in the text.


```python

entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

```

(S
  This/DT
  oatmeal/NN
  is/VBZ
  not/RB
  good/JJ
  ./.
  Its/PRP$
  mushy/NN
  ,/,
  soft/JJ
  ,/,
  I/PRP
  do/VBP
  n't/RB
  like/VB
  it/PRP
  ./.
  (ORGANIZATION Quaker/NNP Oats/NNPS)
  is/VBZ
  the/DT
  way/NN
  to/TO
  go/VB
  ./.)


## Sentiment Analysis Techniques


### VADER Sentiment Scoring

#### Import Libraries and Initialize Sentiment Analyzer

To perform sentiment analysis, we use NLTK's SentimentIntensityAnalyzer (VADER). We initialize the analyzer and process each review to calculate sentiment scores.


```python

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()
sia.polarity_scores(example)

```

{'neg': 0.22, 'neu': 0.78, 'pos': 0.0, 'compound': -0.5448}

**Sentiment Analysis:** The SentimentIntensityAnalyzer calculates the sentiment scores, which include:

**neg:** The proportion of text that expresses negative sentiment.

**neu:** The proportion of text that is neutral.

**pos:** The proportion of text that expresses positive sentiment.

**compound:** A normalized score that combines the above scores into a single value, ranging from -1 (most negative) to 1 (most positive).


This approach provides a simple yet effective way to gauge the sentiment of text data using VADER.

```python

from tqdm import tqdm  # Ensure tqdm is imported if used

# Initialize an empty dictionary to store results
res = {}

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    
    # Check the text and ID to ensure they are not empty or NaN
    if pd.notna(text) and pd.notna(myid):
        res[myid] = sia.polarity_scores(text)
    else:
        print(f"Skipping row {i} due to missing values.")

```



#### Compute Sentiment Scores for All Reviews

We iterate through each review, calculate sentiment scores, and store the results in a dictionary.

```python

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')
vaders

```

#### Visualize Sentiment Scores

We visualize the compound sentiment score, which represents the overall sentiment, by Amazon star rating.

```python

custom_palette = sns.color_palette("pastel")
ax = sns.barplot(data=vaders, x='Score', y='compound', palette=custom_palette)
ax.set_title('Compound Score by Amazon Star Review')
plt.show()

```

<img width="428" alt="image" src="https://github.com/user-attachments/assets/0d00d626-9dce-44a6-89bd-5fcbedd54f1d">


**Creating DataFrame**
​
vaders DataFrame is created from the sentiment results. Each row corresponds to a review, and the sentiment scores (compound, pos, neu, neg) are added.
​
**Merging DataFrames:**
​
vaders is merged with the original df DataFrame to combine the sentiment scores with the review information.
​
**Visualization:**
​
The first plot shows the compound sentiment score (overall sentiment) by review star rating.
The second set of plots shows the positive, neutral, and negative sentiment scores for each review star rating, giving a detailed view of how sentiment varies with the rating.

```python

custom_palette = sns.color_palette("Set2")

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0], palette=custom_palette)
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1], palette=custom_palette)
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2], palette=custom_palette)
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

```

<img width="599" alt="image" src="https://github.com/user-attachments/assets/0c3613f3-dbc6-4e1c-9ad0-084afc696e55">



### Roberta Pretrained Model

Use a model trained of a large corpus of data. Transformer model accounts for the words but also the context related to other words.

```python
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

```

#### Load the Model and Tokenizer

Load the pre-trained model and tokenizer for sentiment analysis.


```python

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

```

```python

# VADER results on example
print(example)
sia.polarity_scores(example)

```

This oatmeal is not good. Its mushy, soft, I don't like it. Quaker Oats is the way to go.
{'neg': 0.22, 'neu': 0.78, 'pos': 0.0, 'compound': -0.5448}


#### Analyze the sentiment of a text using the Roberta model

Here we are goin to analyze the sentiment of a given text using the Roberta model from Hugging Face's transformers library. The process involves tokenizing the text, passing it through the model, and then applying the softmax function to obtain probabilities for negative, neutral, and positive sentiments.

```python

# Run for Roberta Model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)

```
{'roberta_neg': 0.97635514, 'roberta_neu': 0.020687453, 'roberta_pos': 0.0029573678}


The function polarity_scores_roberta is designed to analyze the sentiment of a given text using the Roberta model from Hugging Face's transformers library. The function performs tokenization, model inference, applies softmax to the output, and then returns the sentiment scores in a dictionary format.

This will return the sentiment analysis results for the given text, showing the probabilities for negative, neutral, and positive sentiments according to the Roberta model.

```python

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

```



Here we intended to run sentiment analysis on a dataset of Amazon reviews, combining results from both VADER (a rule-based model) and Roberta (a transformer-based model). It iterates over each row in the dataframe (df), processes the text using both models, and then stores the results in a dictionary (res) keyed by the review ID (myid).

**After Running the Code:**

Once the loop completes, we will have a dictionary res that contains the sentiment scores for each review in our dataset. We can then convert this dictionary into a DataFrame for further analysis, visualization, or modeling.



```python

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')

```


After running this code, we'll have a DataFrame (results_df) that includes all the original columns from df, plus the sentiment scores generated by VADER and Roberta.

```python

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

```

This will output a list of all the column names in results_df.


```python

results_df.columns

```

Index(['Id', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound',
       'roberta_neg', 'roberta_neu', 'roberta_pos', 'ProductId', 'UserId',
       'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator',
       'Score', 'Time', 'Summary', 'Text', 'HelpfulnessRatio', 'ReviewTime',
       'WordCount'],
      dtype='object')

      

### Combine and compare


To visualize the relationships between the VADER and RoBERTa sentiment scores with respect to the review Score, we can use Seaborn's pairplot.

```python

sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='coolwarm')
plt.show()
```

<img width="295" alt="image" src="https://github.com/user-attachments/assets/e259597e-e377-40e1-b2ae-1c4d58d57cf8">




### Review Examples
​
To find examples where the model scoring and review score differ the most, we can query your DataFrame to find reviews where the model's sentiment score and the review's star rating are at odds. Specifically, we can look at extreme cases like positive sentiment in 1-star reviews and negative sentiment in 5-star reviews. Here's how you might approach it:
​
​
Finding Examples of Positive 1-Star and Negative 5-Star Reviews
​
**Positive 1-Star Review:**
​
Find the review with the highest positive sentiment (roberta_pos) score among those with a 1-star rating.
​
**Negative 5-Star Review:**
​
Find the review with the highest negative sentiment (roberta_neg) score among those with a 5-star rating.


This code will generate a grid of scatter plots showing the pairwise relationships between VADER and RoBERTa sentiment scores, with points colored by the review Score.

```python

import pandas as pd

# Find the most positive 1-star review
positive_1_star_reviews = results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)
most_positive_1_star_review = positive_1_star_reviews['Text'].values[0] if not positive_1_star_reviews.empty else "No 1-star reviews found"

# Find the most negative 5-star review
negative_5_star_reviews = results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)
most_negative_5_star_review = negative_5_star_reviews['Text'].values[0] if not negative_5_star_reviews.empty else "No 5-star reviews found"

print("Most Positive 1-Star Review:")
print(most_positive_1_star_review)
print("\nMost Negative 5-Star Review:")
print(most_negative_5_star_review)

```

Most Positive 1-Star Review:
I felt energized within five minutes, but it lasted for about 45 minutes. I paid $3.99 for this drink. I could have just drunk a cup of coffee and saved my money.

Most Negative 5-Star Review:
this was sooooo deliscious but too bad i ate em too fast and gained 2 pds! my fault



### The Transformers Pipeline


Using the pipeline API from the transformers library is a great way to quickly perform sentiment analysis with pre-trained models. The pipeline function simplifies the process of applying a model to text data, allowing us to easily get sentiment scores or labels.


```python

from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")

```

#### Initialize the Pipeline:

**sent_pipeline = pipeline("sentiment-analysis"):**  Creates a sentiment analysis pipeline using a default pre-trained model. By default, it uses a model fine-tuned on sentiment analysis tasks (like distilbert-base-uncased-finetuned-sst-2-english).

#### Perform Sentiment Analysis:

**sent_pipeline(texts):** Applies sentiment analysis to each text in the texts list. The pipeline returns a list of dictionaries with sentiment labels and scores.
Print Results:

Iterates over the texts and their corresponding results, printing out each text’s sentiment and confidence score.



Using the sent_pipeline to analyze the sentiment of a single text is straightforward.


```python

from transformers import pipeline

# Initialize the sentiment analysis pipeline
sent_pipeline = pipeline("sentiment-analysis")

# Analyze the sentiment of a single text
text = "Make sure to like and subscribe!"
result = sent_pipeline(text)

# Print the result
print(result)

```


[{'label': 'POSITIVE', 'score': 0.9991742968559265}]


## Conclusion


In this project, we explored sentiment analysis using two different approaches: VADER, a rule-based model, and Roberta, a transformer-based model. The comparison between these models allowed us to understand the strengths and weaknesses of each method.

The VADER model is efficient for quick sentiment analysis, especially when dealing with social media text or short comments. However, its rule-based nature can sometimes lead to less nuanced results, particularly when the sentiment is context-dependent.

On the other hand, the Roberta model, being a transformer-based model, provides a more sophisticated analysis by considering the context of the words in a sentence. This is evident in its ability to capture more nuanced sentiments in the Amazon reviews dataset.

By combining the results from both models, we were able to perform a comprehensive sentiment analysis, revealing interesting patterns in the data. For instance, the ability to identify positive sentiments in low-rated reviews and negative sentiments in high-rated reviews highlighted the complexity of customer opinions, which might not always align with the given star ratings.

Moreover, we utilized the Transformers pipeline for a more streamlined approach to sentiment analysis, demonstrating the ease of applying pre-trained models to text data. This method not only simplifies the process but also ensures that we can quickly and effectively analyze large datasets with minimal setup.

Overall, this project underscores the importance of using multiple approaches in sentiment analysis, as each method can offer unique insights. By leveraging both rule-based and transformer-based models, we can achieve a more balanced and thorough understanding of textual sentiments, which is crucial for applications in customer feedback analysis, social media monitoring, and beyond.





































