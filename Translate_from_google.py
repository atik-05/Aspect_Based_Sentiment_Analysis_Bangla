# from bs4 import BeautifulSoup
# import requests
#
# str = 'word'
# url = 'https://translate.google.com/#en/bn/' +str
# base_page = requests.get(url)
# soup = BeautifulSoup(base_page.text, 'lxml')
# para = soup.find('div', attrs={'class': 'viewport'})
# print(soup)


from googletrans import Translator
translator = Translator()
print(translator.translate('We were very disappointed.', dest='bn').text)
