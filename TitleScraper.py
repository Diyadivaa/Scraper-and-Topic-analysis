import requests
import csv
from bs4 import BeautifulSoup

URL = "https://steno.ai/lex-fridman-podcast-10"

# Open a CSV file for writing
with open('episodes.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    page = 1
    while True:
        # Make a request to the URL formed by appending the page number to the base URL
        page_url = f"{URL}?page={page}"
        page_response = requests.get(page_url)
        soup = BeautifulSoup(page_response.content, 'html.parser')

        # Extract the episode titles from the response
        episode_links = soup.find_all('a', class_='article__episode tg-subtitle')
        for episode in episode_links:
            writer.writerow([episode.text])

        # Check if the load more button is present on the page
        load_more_button = soup.find(class_='btn btn-black tg-button')
        if not load_more_button:
            break

        # Increment the page number
        page += 1
