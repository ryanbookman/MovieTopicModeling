import pandas as pd
import numpy as np
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import string
import os

from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from bertopic import BERTopic
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

def remove_people_names(text):
  tagged_words = pos_tag(nltk.word_tokenize(text))
  filtered_words = [word for word, pos in tagged_words if pos != 'NNP'] # removes proper nouns
  return ' '.join(filtered_words)

# splits text into sentences of length <= 3
def split_into_paragraphs(text_list, indexingDocs):
    paragraphs = []
    index = 0

    for text in text_list:
        sentences = sent_tokenize(text)
        while len(sentences) > 0:
            if len(sentences) < 3:
                paragraphs.append(' '.join(sentences))
                break
            else:
                paragraphs.append(' '.join(sentences[:3]))
                indexingDocs[paragraphs[-1]] = index
                sentences = sentences[3:]
        index += 1
    return paragraphs

"""
BERTopic Model
"""
def BERTopicModel(docs, reviewsDf):
  # embedding, dimensionality reduction, and clustering
  embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
  umap_model = UMAP(n_neighbors = 25, n_components=5, min_dist=0.0, metric='cosine', random_state=0)
  hdbscan_model = HDBSCAN(min_cluster_size=len(docs)//75, metric='euclidean', cluster_selection_method='leaf', prediction_data=True)

  # create topic representation from the clusters
  vectorizer_model = CountVectorizer(stop_words="english", min_df=2, max_df = len(reviewsDf), ngram_range=(1,2))
  ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
  representation_model = MaximalMarginalRelevance(diversity=0.3, top_n_words=10)

  # define the pipeline
  BERTopic_model = BERTopic(
    embedding_model = embedding_model,
    umap_model = umap_model,
    hdbscan_model = hdbscan_model,
    vectorizer_model = vectorizer_model,
    ctfidf_model = ctfidf_model,
    representation_model = representation_model,
    calculate_probabilities = True,
    top_n_words=10
  )
  return BERTopic_model

"""
reviewsDict: Dictionary with keys (0 - #topics) with list of
all ORIGINAL reviews that have sub-texts belonging to that topic
"""
def documentReversal(topicsDf, reviewsDf, docs, indexingDocs, probs):
  reviewsDict = defaultdict(list)

  for i in range(len(probs)):
    docIndex = np.argmax(probs[i])
    reviewsDict[docIndex].append(reviewsDf["Reviews"][indexingDocs[docs[i]]])

  return reviewsDict

"""
Sentiment Analysis
"""
def sentimentSegmentation(neg, neu, pos, reviewsDict):
  sentiment_for_each_topic = []
  sia = SentimentIntensityAnalyzer()

  for i in range(0, len(reviewsDict) - 1):
    total_score_for_topic = 0
    total_count_for_topic = 0

    for text in reviewsDict[i]:
      sentiment_score = sia.polarity_scores(text)['compound']
      if (sentiment_score <= -0.25):
        neg[i] += 1
      elif (sentiment_score >= 0.25):
        pos[i] += 1
      else:
        neu[i] += 1

def generate_labels(topicsDf):
  client = OpenAI(api_key='sk-APIKEYHERExxx')

  INSTRUCTIONS = """You are a topic labeler for a movie review segmentation model. The developers have already clustered movie reviews into categories, and each category was assigned a list of the most common words (called 'Representation') that represent it. You will be given this list of representative words along with a few example reviews (called 'Representative_Docs') from that category.

Your task is to output a single, concise category label that best describes the given representative words and example reviews, regardless of the sentiment expressed in the reviews. The label should be a short phrase or a few words, such as 'Music', 'Acting', 'Story Arc', 'Math/Professor Elements', 'Character Depth', 'Emotional Engagement', or 'Messaging/Morals'.

The input will be provided in the following format:

Here are the representative words for this cluster:
[list of representative words separated by commas or newlines]

Here are some representative reviews:
[multiple example reviews separated by newlines]

Give a label for this segmented category.

Please respond with only the category label, without any additional explanation or context. If you cannot determine a suitable label from the given information, respond with 'Unclear'. When determining the category label, focus on the topics or aspects of the movie being discussed, rather than the overall sentiment or opinion expressed in the reviews."""

  responses = []
  messages = [
      {"role": "system", "content": INSTRUCTIONS}
      ]

  for index, row in topicsDf.iterrows():
    messages.append(
        {"role": "user", "content": f"Here are the representative words for this cluster:\n\n{topicsDf['Representation'].iloc[index]}\n\nHere are some representative reviews:{topicsDf['Representative_Docs'].iloc[index][:3]}. Give a label for this segmented category."},
    )
    response = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=messages
    )
    responses.append(response.choices[0].message.content)
    messages.append(
        {"role": "assistant", "content": response.choices[0].message.content},
    )

  return responses


def preprocess_embeddings(labels, topicsDf):
  keywordsList = []

  for index, row in topicsDf.iterrows():
    key_words = topicsDf['Representation'].iloc[index][:10]
    keywordsList.append([labels[index - 1]] + key_words)

  return keywordsList


def embed(labels, topicsDf):
  client = OpenAI(api_key='sk-APIKEYHERExxx')

  embeddings = []

  for keywords in preprocess_embeddings(labels, topicsDf):
    response = client.embeddings.create(
      input=keywords,
      model="text-embedding-3-small"
  )

    embeddings.append(response.data[0].embedding)
  return embeddings


def calcSimilarities(embeddings):
  similarity_threshold = 0.5

  embeddings = np.array(embeddings)
  similarities = cosine_similarity(embeddings)
  similaritiesArr = []

  for i in range(len(similarities)):
      for j in range(i + 1, len(similarities)):
          similarity = similarities[i, j]
          if similarity > similarity_threshold:
            similaritiesArr.append([i, j, similarity])

  similaritiesArr = np.array(similaritiesArr)
  similaritiesArrSorted = []

  if (len(similaritiesArr) > 0):
    similaritiesArrSorted = similaritiesArr[similaritiesArr[:, 2].argsort()[::-1]]
  else:
    similaritiesArrSorted = similaritiesArrSorted

  return similaritiesArrSorted


def relabelling(similaritiesArrSorted, neg, neu, pos, labels):
  for ele in similaritiesArrSorted:
    if (len(labels) <= 5):
      break

    neg[int(ele[0])] += neg[int(ele[1])]
    neu[int(ele[0])] += neu[int(ele[1])]
    pos[int(ele[0])] += pos[int(ele[1])]

    neg[int(ele[1])] = 0
    neu[int(ele[1])] = 0
    pos[int(ele[1])] = 0

  index = 0
  while (index < len(neg)):
    if neg[index] == 0 and neu[index] == 0 and pos[index] == 0:
      del neg[index]
      del neu[index]
      del pos[index]
      del labels[index]
      index -= 1
    index += 1

  for i in range(len(neg)):
    tot = neg[i]+neu[i]+pos[i]
    neg[i] = neg[i]/tot
    neu[i] = neu[i]/tot
    pos[i] = pos[i]/tot

