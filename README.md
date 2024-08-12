# SentimentAnalysis-AmazonReviews
Sentiment Analysis of Amazon Customer Reviews Using NLTK and VADER



## Table of Contents

1. [Introduction](#introduction)
2. [Data Source](#data-source)
3. [Methodology](#methodology)
4. [Visualizations](#visualizations)
5. [Dashboard](#dashboard)
6. [Discussion](#discussion)
7. [Conclusion](#Conclusion)




## Introduction

This project involves sentiment analysis on Amazon customer reviews using Natural Language Processing (NLP) techniques. The goal is to gain insights into the sentiment behind the reviews by analyzing their text content, performing sentiment scoring, and visualizing the results. This project utilizes the Natural Language Toolkit (NLTK) for basic NLP operations, as well as VADER (Valence Aware Dictionary and sEntiment Reasoner) for sentiment analysis.


## Data Source 




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

## Exploratory Data Analysis (EDA)

**Count of Reviews by Stars**

We begin our exploratory data analysis by visualizing the count of reviews for each star rating in the dataset.

```python

ax = df['Score'].value_counts().sort_index().plot(kind='bar', color='#3CB371', title='Count of Reviews by Stars', figsize=(10, 5))
ax.set_xlabel('Review Stars')
ax.set_ylabel('Number of Reviews')
plt.show()

```


**Helpfulness Ratio Analysis**

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

**Review Count by Time**

We convert the review time from UNIX timestamp to a datetime format and analyze the number of reviews over time.

```python

df['ReviewTime'] = pd.to_datetime(df['Time'], unit='s')
df.set_index('ReviewTime')['Id'].resample('M').count().plot(figsize=(10, 5), color='teal')
plt.title('Number of Reviews Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Reviews')
plt.show()


```

## Distribution of Word Count in Reviews

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

## Basic NLTK Operations


**Tokenization**


We tokenize the text in the reviews, breaking it down into individual words.


```python

example = df['Text'][50]
tokens = nltk.word_tokenize(example)
tokens[:10]

```


**Part-of-Speech Tagging**

We then tag each token with its part of speech (POS) to understand the grammatical structure of the text.

```python

tagged = nltk.pos_tag(tokens)
tagged[:10]

```


**Named Entity Recognition (NER)**

Finally, we perform named entity recognition to identify entities such as people, organizations, and locations in the text.


```python

entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

```


## VADER Sentiment Scoring

**Import Libraries and Initialize Sentiment Analyzer**

To perform sentiment analysis, we use NLTK's SentimentIntensityAnalyzer (VADER). We initialize the analyzer and process each review to calculate sentiment scores.


```python

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()
sia.polarity_scores(example)

```

**Compute Sentiment Scores for All Reviews**

We iterate through each review, calculate sentiment scores, and store the results in a dictionary.

```python

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)


```


**Create DataFrame and Merge with Original Data**

We create a DataFrame from the sentiment scores and merge it with the original review data to combine the sentiment scores with other review details.


```python

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')
vaders.head()

```

## Visualize Sentiment Scores


**Compound Score by Amazon Star Review**

We visualize the compound sentiment score, which represents the overall sentiment, by Amazon star rating.

```python

custom_palette = sns.color_palette("pastel")
ax = sns.barplot(data=vaders, x='Score', y='compound', palette=custom_palette)
ax.set_title('Compound Score by Amazon Star Review')
plt.show()

```

**Positive, Neutral, and Negative Scores by Amazon Star Review**

We further break down the sentiment scores into positive, neutral, and negative components and visualize them by star rating.


```python

custom_palette = sns.color_palette("Set2")
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0], palette=custom_palette)
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1], palette=custom_palette)
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2], palette=custom_palette)
plt.tight_layout()
plt.show()

```


## Roberta Pretrained Model


Use a model trained of a large corpus of data. Transformer model accounts for the words but also the context related to other words.

```python

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

```

**Load the Model and Tokenizer:**

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


**Analyze Sentiment Using Roberta**

To analyze the sentiment of a text using the Roberta model, we will tokenize the text, pass it through the model, and then apply the softmax function to obtain probabilities for negative, neutral, and positive sentiments.


```python

# Run for Roberta Model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg': scores[0],
    'roberta_neu': scores[1],
    'roberta_pos': scores[2]
}
print(scores_dict)

```

**Function to Get Roberta Sentiment Scores**

This function, polarity_scores_roberta, analyzes the sentiment of a given text using the Roberta model, returning the sentiment scores in a dictionary format.

```python

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

```

**Running Sentiment Analysis on Amazon Reviews Dataset**

We run sentiment analysis on a dataset of Amazon reviews, combining results from both VADER and Roberta models. This code iterates over each row in the dataset, processes the text using both models, and stores the results in a dictionary.


```python

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {f"vader_{key}": value for key, value in vader_result.items()}
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')


```

**Creating a DataFrame with Sentiment Scores**

After running the analysis, we create a DataFrame results_df that includes all the original columns from the dataset, plus the sentiment scores generated by VADER and Roberta.

```python

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

```


## The Transformers Pipeline

Using the pipeline API from the transformers library is a quick way to perform sentiment analysis with pre-trained models. The pipeline function simplifies the process of applying a model to text data.

```python

from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")

```


**Performing Sentiment Analysis**

We analyze the sentiment of a single text using the sent_pipeline.


```python

# Analyze the sentiment of a single text
text = "Make sure to like and subscribe!"
result = sent_pipeline(text)

# Print the result
print(result)

```


**Visualizing the Sentiment Scores**

To visualize the relationships between VADER and Roberta sentiment scores with respect to the review Score, we use Seaborn's pairplot.


```python

sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                   'roberta_neg', 'roberta_neu', 'roberta_pos'],
             hue='Score',
             palette='coolwarm')
plt.show()


```


## Review Examples

We find examples where the model scoring and review score differ the most, specifically looking at cases like positive sentiment in 1-star reviews and negative sentiment in 5-star reviews.

```python

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




## Conclusion

Conclusion
In this project, we explored sentiment analysis using two different approaches: VADER, a rule-based model, and Roberta, a transformer-based model. The comparison between these models allowed us to understand the strengths and weaknesses of each method.

The VADER model is efficient for quick sentiment analysis, especially when dealing with social media text or short comments. However, its rule-based nature can sometimes lead to less nuanced results, particularly when the sentiment is context-dependent.

On the other hand, the Roberta model, being a transformer-based model, provides a more sophisticated analysis by considering the context of the words in a sentence. This is evident in its ability to capture more nuanced sentiments in the Amazon reviews dataset.

By combining the results from both models, we were able to perform a comprehensive sentiment analysis, revealing interesting patterns in the data. For instance, the ability to identify positive sentiments in low-rated reviews and negative sentiments in high-rated reviews highlighted the complexity of customer opinions, which might not always align with the given star ratings.

Moreover, we utilized the Transformers pipeline for a more streamlined approach to sentiment analysis, demonstrating the ease of applying pre-trained models to text data. This method not only simplifies the process but also ensures that we can quickly and effectively analyze large datasets with minimal setup.

Overall, this project underscores the importance of using multiple approaches in sentiment analysis, as each method can offer unique insights. By leveraging both rule-based and transformer-based models, we can achieve a more balanced and thorough understanding of textual sentiments, which is crucial for applications in customer feedback analysis, social media monitoring, and beyond.





































