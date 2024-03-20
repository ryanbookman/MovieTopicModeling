from flask import Flask, render_template, request, jsonify
from scrape import get_movie_info
from threading import Thread
from ml import *
import shutil

app = Flask(__name__)

reviews = []
plot_image_path = '../static/plot.png'

@app.route('/process', methods=['POST'])
def process():
    global plot_image_path
    global reviews
    querie = request.form['querie']
    get_movie_info(querie, reviews)
    reviews_length = len(reviews)

    thread = Thread(target=runML)
    thread.start()

    return jsonify({'reviews_length': reviews_length})

def runML():
  global reviews
  global plot_image_path
  print("STARTING ANALYSIS")
  reviewsDf = pd.DataFrame({"Reviews": reviews})
  reviewsDf['Reviews'] = reviewsDf['Reviews'].apply(lambda x: remove_people_names(x))

  indexingDocs = defaultdict(int)
  # docs is the new corpus to be inputted into BERTopic model
  docs = split_into_paragraphs(reviewsDf['Reviews'], indexingDocs)

  """
  topics stores which topic each 'document' in the corpus belongs to
  probs stores the probability of each 'document' belonging to all categories
  -1 label for a topic are attributed to 'documents' that don't fit well with topics
  """
  BERTopic_model = BERTopicModel(docs, reviewsDf)
  topics, probs = BERTopic_model.fit_transform(docs)

  topicsDf = BERTopic_model.get_topic_info() # DataFrame with information about each topic
  topicsDf = topicsDf.loc[1:].reset_index(drop=True)

  # clearning topicsDf for categories with < 10 representative keywords
  for i in range(len(topicsDf)):
    if topicsDf["Representation"][i][-1] == '':
      topicsDf = topicsDf.drop(i)
      probs = np.delete(probs, i, axis = 1)
  topicsDf = topicsDf.reset_index(drop=True)

  # sentiment evaluation for each topic
  reviewsDict = documentReversal(topicsDf, reviewsDf, docs, indexingDocs, probs)
  neg = [0]*(len(reviewsDict))
  neu = [0]*(len(reviewsDict))
  pos = [0]*(len(reviewsDict))
  sentimentSegmentation(neg, neu, pos, reviewsDict)

  # initial topic labels
  labels = generate_labels(topicsDf)

  # fine tuning topic labels
  embeddings = embed(labels, topicsDf)
  similaritiesArrSorted = calcSimilarities(embeddings)
  relabelling(similaritiesArrSorted, neg, neu, pos, labels)

  finalDf = pd.DataFrame({"Topics": labels, "Negative": neg, "Neutral": neu, "Positive": pos})

  # deleting rows with "Unclear" label
  for i in range(len(finalDf)):
    if (finalDf.loc[i]["Topics"] == "Unclear"):
      del neg[i]
      del neu[i]
      del pos[i]
  finalDf = finalDf[finalDf["Topics"] != "Unclear"].reset_index(drop=True)

  # plotting
  fig = px.bar(finalDf, y="Topics", x=["Negative", "Neutral", "Positive"],
              title="Sentiment Distribution by Topic",
              labels={"value": "Sentiment Distribution", "Topics": "Topics"},
              color_discrete_map={"Negative": "red", "Neutral": "gray", "Positive": "green"},
            orientation="h")
  
  image_path = os.path.join(os.getcwd(), 'static', 'plot.png')
  fig.write_image(image_path)

  #reset reviews after using it to the ML application
  reviews = []
  
  print('IMAGE GENERATED')
  return plot_image_path

@app.route('/get_image_path', methods=['GET'])
def get_image_path():
    return jsonify({'plot_image_path': plot_image_path})

@app.route('/', methods=['GET'])
def index():
      shutil.copyfile('static/preset.png', 'static/plot.png')
      return render_template('index.html', plot_image_path=plot_image_path)

if __name__ == '__main__':
    app.run(debug=True)