# ---- Method 1 ----

# import html2text
# import requests

# # Fetch HTML content
# url = "https://medium.com/elysia-magazine/elysia-ecosystem-zkrwa-52e578f9f2be"
# response = requests.get(url)
# html_content = response.text

# # Convert HTML to text
# text_maker = html2text.HTML2Text()
# text_maker.ignore_links = True  # Optional: ignore links
# text_maker.ignore_images = True  # Optional: ignore images

# text = text_maker.handle(html_content)
# print(text)

# ---- Method 2 ----

# from lxml import html
# import requests

# # Fetch HTML content
# url = "https://medium.com/elysia-magazine/elysia-ecosystem-zkrwa-52e578f9f2be"
# response = requests.get(url)
# html_content = response.content

# # Parse HTML and extract text
# tree = html.fromstring(html_content)
# text = tree.xpath('//text()')  # Extract all text nodes
# cleaned_text = ' '.join(text).strip()
# print(cleaned_text)


# ---- Method 3 ----

# from goose3 import Goose

# # Initialize Goose extractor
# g = Goose()

# # Extract article content
# # url = "https://medium.com/elysia-magazine/elysia-ecosystem-zkrwa-52e578f9f2be"
# url = "https://news.ycombinator.com/"
# article = g.extract(url=url)

# # Get textual content
# print(article.cleaned_text)

# ---- Method 4 ----

from bs4 import BeautifulSoup
import requests

# Fetch HTML content
url = "https://medium.com/elysia-magazine/elysia-ecosystem-zkrwa-52e578f9f2be"
response = requests.get(url)
html_content = response.text

# Parse HTML content
soup = BeautifulSoup(html_content, "html.parser")

# Extract only the textual content
text = soup.get_text()

# Optional: Clean up text (remove extra whitespaces, newlines)
cleaned_text = ' '.join(text.split())
print(cleaned_text)
