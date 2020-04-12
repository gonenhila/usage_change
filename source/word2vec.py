# run word2vec on custom dataset with two different seeds

import gensim
import sys
import codecs
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data", 
                    default='usage_change/tokdata/birthyear.1990_2009.lowercase', 
                    help="name of tokenized data file")
parser.add_argument("--save", 
                    default='/tmp/old', 
                    help="prefix for directory to save the .wv.npy, .vocab")

def run_and_save(args, seed):
  # train word2vec model, save word embeddings (.wv.npy) and vocab file (.vocab)

  # train word2vec model
  print('training %s for seed %d'%(args.data, seed))
  model = gensim.models.Word2Vec(size=300, alpha=0.025, window=4, min_count=20, seed=seed, workers=12, sg=1, hs=0, negative=5, iter=5)
  model.build_vocab(gensim.models.word2vec.LineSentence(args.data))
  model.train(gensim.models.word2vec.LineSentence(args.data), total_examples=model.corpus_count, epochs=model.iter)
  save_prefix = args.save + '.seed%d'%(seed)
  model.wv.save_word2vec_format(save_prefix + '.tmp')

  # save .wv.npy and .vocab
  vec = []
  w = codecs.open(save_prefix + '.vocab', 'w', encoding='utf-8')
  vocab_size, embed_dim = None, None
  with codecs.open(save_prefix + '.tmp', 'r', encoding='utf-8', errors='ignore') as r:
    for line in r:
      items = line.strip().split()
      if not vocab_size:
        assert(len(items) == 2)
        vocab_size, embed_dim = int(items[0]), int(items[1])
      else:
        assert(len(items) == embed_dim + 1)
        vec.append([float(item) for item in items[1:]])
        w.write('%s\n'%items[0])
  w.close()
  vec = np.array(vec, dtype=np.float)
  assert(vec.shape[0] == vocab_size)
  assert(vec.shape[1] == embed_dim)
  np.save(save_prefix + '.wv.npy', vec)
  print('saved %s.wv.npy'%save_prefix)
  print('saved %s.vocab'%save_prefix)

  os.remove(save_prefix + '.tmp')

if __name__ == '__main__':
  args = parser.parse_args()
  run_and_save(args, seed=123)
  run_and_save(args, seed=456)


