# MovieTopicModeling
Repo with custom implementation of end-to-end IMDb scraping, topic-modeling, sentiment analysis and website integration via Flask The application allows users to enter a movie title, and it scrapes the reviews for that movie from IMDb. The scraped reviews are then analyzed using the BERTopic model, we process these results with OpenAI and remove redundancies with by looking at cosine similarity, and the sentiment distribution across different topics is visualized in a bar chart.

**Features**
Scrape movie reviews from IMDb using Selenium
Preprocess reviews by removing proper nouns
Cluster reviews into topics using the BERTopic model
Perform sentiment analysis on the reviews for each topic
Generate topic labels using OpenAI's GPT-3
Fine-tune topic labels based on their semantic similarity
Visualize sentiment distribution across topics in a bar chart

**Installation and Setup**
Clone the repository:

Copy code
1. git clone https://github.com/ryanbookman/MovieTopicModeling.git
2. Install the required dependencies.
3. Add your OpenAI token in app.py.

Download the ChromeDriver executable and add it to your system's PATH.

**Usage**
Start the Flask application:

Copy code
python app.py
Open your web browser and navigate to http://localhost:5000.
Enter a movie title in the search box and click "Submit".
Wait for the application to scrape the reviews and perform the analysis.
The sentiment distribution across different topics will be visualized in a bar chart.

**Code Structure**
scrape.py: Contains functions for scraping movie reviews from IMDb using Selenium.
ml.py: Contains functions for preprocessing reviews, clustering them into topics using BERTopic, performing sentiment analysis, generating topic labels, and visualizing the sentiment distribution.
app.py: The Flask application that handles the web interface and coordinates the scraping and analysis processes.
templates/index.html: The HTML template for the web interface.

**Dependencies**
The project relies on the following major dependencies:

Flask: A lightweight Python web framework.
Selenium: A web browser automation tool for scraping data.
NLTK: A natural language processing toolkit for text preprocessing.
BERTopic: A topic modeling library based on BERT and c-TF-IDF.
OpenAI: An API for accessing OpenAI's language models, used for generating topic labels.
Plotly: A data visualization library for creating interactive plots.
For a complete list of dependencies, see code imports.


**Acknowledgments**
The BERTopic library: https://github.com/MaartenGr/BERTopic
OpenAI's GPT-3 language model
Plotly for data visualization
