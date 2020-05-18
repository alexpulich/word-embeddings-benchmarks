class Embedding:
    def __init__(self, fname, binary):
        self.embeddings = Embedding.from_word2vec(fname, binary=binary)


    @staticmethod
    def _from_word2vec_text(fname):
        with open(fname, 'r') as fin:
            words = []

            header = fin.readline()
            ignored = 0
            vocab_size, layer1_size = list(map(int, header.split()))  # throws for invalid file format
            vectors = np.zeros(shape=(vocab_size, layer1_size), dtype=np.float32)
            for line_no, line in enumerate(fin):
                try:
                    parts = text_type(line, encoding="utf-8").split(' ')
                    w = parts[0]
                    parts = list(map(lambda x: x.strip(), parts[1:]))
                    parts.insert(0, w)

                except TypeError as e:
                    parts = line.split(' ')
                    w = parts[0]
                    parts = list(map(lambda x: x.strip(), parts[1:]))
                    parts.insert(0, w)

                except Exception as e:
                    logger.warning("We ignored line number {} because of errors in parsing"
                                   "\n{}".format(line_no, e))
                    continue

                # We differ from Gensim implementation.
                # Our assumption that a difference of one happens because of having a
                # space in the word.
                if len(parts) == layer1_size + 1:
                    word, vectors[line_no - ignored] = parts[0], list(map(np.float32, parts[1:]))
                elif len(parts) == layer1_size + 2 and parts[-1]:
                    # last element after splitting is not empty- some glove corpora have additional space
                    word, vectors[line_no - ignored] = parts[:2], list(map(np.float32, parts[2:]))
                    word = u" ".join(word)
                elif not parts[-1]:
                    # omit last value - empty string
                    word, vectors[line_no - ignored] = parts[0], list(map(np.float32, parts[1:-1]))
                else:
                    ignored += 1
                    logger.warning("We ignored line number {} because of unrecognized "
                                   "number of columns {}".format(line_no, parts[:-layer1_size]))
                    continue

                words.append(word)

            if ignored:
                vectors = vectors[0:-ignored]

            if len(words) < vocab_size:
                logger.warning("Omitted {} words".format(vocab_size - len(words)))
            elif len(words) > vocab_size:
                raise RuntimeError("Read too many words, incorrect file")

            return words, vectors

