import random

fin = 'processed_news.txt'
f = open(fin).readlines()
random.shuffle(f)
with open('nyt.txt', 'w') as fout:
    fout.writelines(f)