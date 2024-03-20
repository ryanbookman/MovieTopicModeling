import pandas as pd
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import time
import random
import numpy as np

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_movie_info(querie, reviews):

  querieURL = "https://www.imdb.com/find?q=" + querie

  driver = webdriver.Chrome()
  driver.get(querieURL)
  movie_url = ""
  try:
      movie = WebDriverWait(driver, 10).until(
          EC.presence_of_element_located((By.CLASS_NAME, 'ipc-metadata-list-summary-item__t'))
      )
      movie_url = movie.get_attribute("href")
  finally:
      pass

  movie_url = "reviews/?".join(movie_url.split('?'))

  # extracting reviews
  driver.get(movie_url)

  limit = 0 # limiting number of reviews (# clicks to load more button) for the sake of runtime
  while limit < 10:
    try:
        load_more_button = driver.find_element(By.CSS_SELECTOR, ".ipl-load-more__button")
        load_more_button.click()
        time.sleep(float(np.random.randint(30,40)) / 10)
    except:
        break
    limit += 1

  html = driver.page_source
  soup = BeautifulSoup(html, "html.parser")

  # two different classes for texts based on IMDb's website structure
  for review in soup.find_all("div", class_="text show-more__control"):
    reviews.append(review.get_text()) # reviews array is pass by reference
  for review in soup.find_all("div", class_="text show-more__control clickable"):
    reviews.append(review.get_text())