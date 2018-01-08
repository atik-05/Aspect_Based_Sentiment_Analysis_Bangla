from bs4 import BeautifulSoup
import requests
import logging
from datetime import datetime


f = open('paper.txt', 'a', encoding='utf-8')
base_url = 'http://www.prothom-alo.com'

def save_description(link):
    page = requests.get(link)
    soup = BeautifulSoup(page.text, 'lxml')
    para = soup.find('div', attrs={'class': 'viewport'})
    if para != None:
        para = para.text.strip()
        f.write(para)
        f.write('\n')


link_traversed = 0
prev = '0'
logging.basicConfig(filename='log.txt', level=logging.INFO)

for month in range(9, 13):
    for day in range(1, 29):
        for page_number in range(1, 16):
            all_links = []
            url = 'http://www.prothom-alo.com/archive/2016-%d-%d?page=%d' % (month, day, page_number)
            base_page = requests.get(url)
            soup = BeautifulSoup(base_page.text, 'lxml')

            page = soup.find('div', attrs={'class': 'listing'})  # earlier have 'summery_view'
            if page == None:
                continue
            for link in page.find_all('a'):
                link = link.get('href')
                if "article" in link:
                    full_link = base_url + link

                    # checking for duplicate link
                    ind = full_link.find('article')
                    start = ind + 8
                    last = full_link[start:]  # because id index may very 6/7
                    id_ind = last.find('/')
                    id = last[:id_ind]

                    if id == prev:
                        # print('duplicate link found')
                        continue
                    prev = id

                    print(full_link)
                    save_description(full_link)
                    link_traversed += 1

        logging.info('%s , month: %d, day: %d' % ( datetime.now(), month, day))


print(' total link saved: ', link_traversed)
f.close()


print('...................')
print('...................')

















