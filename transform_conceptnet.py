i = 0
with open('/Users/alexpulich/wrk/uni/nlp/numberbatch.txt', 'r') as fi:
    with open('/Users/alexpulich/wrk/uni/nlp/numberbatch_wo_prefix.txt', 'w') as fo:
        for line in fi:
            if line.startswith('/c/th/'):
                saving_line = line.replace('/c/th/', '')
                fo.write(saving_line)
                i += 1
                print(i)
