import csv
import requests
import time
import random
import pickle

BASE_URL = 'http://api.conceptnet.io/relatedness?node1=/c/%s/%s&node2=/c/%s/%s'
base_dir = '../thai_word_similarity/'
FILES = (
    'th-semeval-500.csv',
    'th-simlex-999.csv',
    'th-wordsim-353.csv',
    'tws65.csv'
)


def get_similarity(word1, word2, lang='th'):
    score = requests.get(BASE_URL % (lang, word1, lang, word2)).json()['value']
    print(f'score for {word1} and {word2} = {score}')
    return score if score > 0 else None


if __name__ == '__main__':
    '''
    scores = {}
    for file in (FILES[3],):
        with open(base_dir + file) as f:
            reader = csv.reader(f, delimiter=',')
            for line in reader:
                if (line[0] + line[1]) not in scores:
                    try:
                        scores[line[0] + line[1]] = get_similarity(line[0], line[1])
                    except Exception as e:
                        print(e)
                        print('waiting for 60 sesc')
                        time.sleep(60)
                        scores[line[0] + line[1]] = get_similarity(line[0], line[1])

    with open('conceptnet.tws.pickle', 'wb') as f:
        pickle.dump(scores, f)
    
    with open('conceptnet.semeval.pickle', 'rb') as f:
        res = pickle.load(f)

    with open('conceptnet.simlex.pickle', 'rb') as f:
        res.update(pickle.load(f))

    with open('conceptnet.tws.pickle', 'rb') as f:
        res.update(pickle.load(f))

    with open('conceptnet.wordsim.pickle', 'rb') as f:
        res.update(pickle.load(f))

    with open('conceptnet.pickle', 'wb') as f:
        pickle.dump(res, f)
    '''
    with open('conceptnet.wordsim.pickle', 'rb') as f:
        ws = pickle.load(f)
    with open(base_dir + FILES[2]) as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            if (line[0] + line[1]) in ws:
                continue
            try:
                ws[line[0] + line[1]] = get_similarity(line[0], line[1])
            except Exception as e:
                print(e)
                print('waiting for 60 sesc')
                time.sleep(60)
                ws[line[0] + line[1]] = get_similarity(line[0], line[1])
            time.sleep(random.randint(50, 60))

    with open('conceptnet.wordsim.pickle', 'wb') as f:
        pickle.dump(ws, f)