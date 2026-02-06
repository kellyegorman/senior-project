# imports needed to get text from urls
import requests
from bs4 import BeautifulSoup
import requests
from bs4 import BeautifulSoup

url = "https://www.bbcgoodfood.com/recipes/collection/easy-baking-recipes"  

# simple function to print the text from a url, using requests and BeautifulSoup to scrape the page content and extract the text
def get_url_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        page_text = soup.get_text(separator='\n', strip=True)
        return page_text
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
    
print(get_url_text(url)[:400])