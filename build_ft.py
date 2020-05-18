"""
Script to create KV model from binary fasttext with required words, but with sub-words encoding feature
"""
import csv
import sys
import fasttext

base_dir = '../thai_word_similarity/'
FILES = (
    'th-semeval-500.csv',
    'th-simlex-999.csv',
    'th-wordsim-353.csv',
    'tws65.csv'
)

if __name__ == '__main__':
    model_path = sys.argv[1]
    word_set = set()
    for file in FILES:
        with open(base_dir + file) as f:
            reader = csv.reader(f, delimiter=',')
            for line in reader:
                word_set.add(line[0])
                word_set.add(line[1])

    model = fasttext.load_model(model_path)
    lines = [str(len(word_set)) + ' 300\n']
    for word in word_set:
        lines.append(word + ' ' + ' '.join((str(w) for w in model[word].tolist())) + '\n')

    with open('model.mc5.kv', 'w') as f:
        f.writelines(lines)