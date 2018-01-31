import pandas as pd
import xlsxwriter
from googletrans import Translator

translator = Translator()


review_dataset = 'data/Restaurant_test.csv'
bangla_review = 'data/bangla_restaurant_test.xlsx'

reviews = pd.read_csv(review_dataset, encoding='ISO-8859-1')
english = reviews['text'].values
category = reviews['category'].values
bangla = []

text_len = len(english)

for i in range(0, text_len):
    en = english[i]
    text = translator.translate(en, dest='bn').text
    bangla.append(text)
    if i == 100:
        print(i)

data = [10,20,30]
cat = [1,0,1]
data_frame = pd.DataFrame({'english': english, 'bangla': bangla, 'category': category})
writer = pd.ExcelWriter(bangla_review, engine='xlsxwriter')
data_frame.to_excel(writer, sheet_name='Sheet1')
writer.save()

print('its done ... ...')
