# use t-SNE to visualize the neighbors of interesting words

import sys
import argparse
from sklearn.manifold import TSNE
import numpy as np
from numpy import linalg as LA
import scipy
from collections import defaultdict
import matplotlib.pyplot as plt
import random
random.seed(123)

parser = argparse.ArgumentParser()
parser.add_argument("--property", default='gender', help="name of split to use")
parser.add_argument("--words", default='pearl', help="interesting words to plot in csv format")
parser.add_argument("--out_dir", default='../data/plots/', help="directory to save the plots")
parser.add_argument("--k", default=50, help="k of k-NN to use")

def extract_freqs(filename, vocab):
  # raw counts
  print('extracting freqs ...')
  count = defaultdict(int)
  with open(filename, 'r') as f:
     for l in f:
        for w in l.strip().split():
          count[w] += 1

  # consider only words in the vocabulary
  count_vocab = defaultdict(int)
  for w in vocab:
    if w in count:
      count_vocab[w] = count[w]

  # normalized frequencies
  tot = sum([count_vocab[item] for item in count_vocab])
  freq_norm = defaultdict(int)
  for w in count_vocab:
    freq_norm[w] = count_vocab[w] / float(tot)

  print('done')
  return freq_norm, count_vocab

def normalize(wv):
  # normalize vectors
  norms = np.apply_along_axis(LA.norm, 1, wv)
  wv = wv / norms[:, np.newaxis]
  return wv

def load_embeddings_from_np(filename):
  print('loading...')
  with open(filename + '.vocab', 'r') as f_embed:
    vocab = [line.strip() for line in f_embed]
  w2i = {w: i for i, w in enumerate(vocab)}
  wv = np.load(filename + '.wv.npy')
  return vocab, wv, w2i

def load_and_normalize(idi, filename, vocab, wv, w2i, hamilton=False):
  cur_vocab, cur_wv, cur_w2i = load_embeddings_from_np(filename)
  cur_wv = normalize(cur_wv)
  vocab[idi] = cur_vocab
  wv[idi] = cur_wv
  w2i[idi] = cur_w2i
  print('loaded and normalized %s embeddings'%idi)

def topK(w, space, k=10, count = None, min_freq = 0):
  # extract the word vector for word w
  idx = w2i[space][w]
  vec = wv[space][idx, :]
  # compute similarity of w with all words in the vocabulary
  sim = wv[space].dot(vec)
  # sort similarities by descending order
  sort_sim = (sim.argsort())[::-1]
  # choose topK
  if count:
    best = []
    for i in sort_sim:
      if i != idx and count[vocab[space][i]] > min_freq:
        best.append(i)
        if len(best) == k:
          break
  else:
     best = sort_sim[:(k + 1)]
  return [vocab[space][i] for i in best if i != idx]

def create_aligned(w2i, wv, vocab, Q_bef, space1, aligned):
  wv[aligned] = np.zeros(wv[space1].shape)

  for i, vec in enumerate(wv[space1]):
    wv[aligned][i, :] = np.dot(vec, Q_bef)

  vocab[aligned] = vocab[space1]
  w2i[aligned] = w2i[space1]


def align(w2i, wv, vocab, space1, space2, aligned):
  train_words = list(set(vocab[space1]).intersection(set(vocab[space2])))
  
  num = len(train_words)
  mat_bef = np.zeros((num, 300))
  mat_aft = np.zeros((num, 300))

  for i, w in enumerate(train_words):
    mat_bef[i, :] = wv[space1][w2i[space1][w]]
    mat_aft[i, :] = wv[space2][w2i[space2][w]]

  Q_bef, s_bef = scipy.linalg.orthogonal_procrustes(mat_bef, mat_aft)

  create_aligned(w2i, wv, vocab, Q_bef, space1, aligned)
  wv[aligned] = normalize(wv[aligned])

def load_all_embeddings(property, val1, val2):
  vocab = {}
  wv = {}
  w2i = {}
  load_and_normalize(val1, '../data/embeddings/{}.{}.lowercase.seed123.mfreq20'.format(property, val1), vocab, wv, w2i)
  load_and_normalize(val2, '../data/embeddings/{}.{}.lowercase.seed123.mfreq20'.format(property, val2), vocab, wv, w2i)
  align(w2i, wv, vocab, val1, val2, val1+'_a')
  return vocab, wv, w2i

def tsne_plot(property, val1, val2):
  # extract frequencies
  freq_norm_val1, count_vocab_val1 = extract_freqs(
      '../data/embeddings/freqs/{}.{}.lowercase'.format(property, val1.split('_')[0]), vocab[val1])
  freq_norm_val2, count_vocab_val2 = extract_freqs(
      '../data/embeddings/freqs/{}.{}.lowercase'.format(property, val2), vocab[val2])

  for int_word in args.words.strip().split(","):
    if int_word in vocab[val1] and int_word in vocab[val2]:
      neighbors_a = set(topK(int_word, val1, args.k, None, 100))
      neighbors_b = set(topK(int_word, val2, args.k, None, 100))
      total_neighbors = neighbors_a.union(neighbors_b)
      neighbor2color = {int_word:'green'}
      for neighbor in total_neighbors:
        if neighbor in neighbors_a and neighbor in neighbors_b:
          neighbor2color[neighbor] = 'purple'
        elif neighbor in neighbors_a:
          neighbor2color[neighbor] = 'blue'
        else:
          neighbor2color[neighbor] = 'red'
      for val in [val1, val2]:
        # construct embedding matrix for neighboring words
        X, wname, colors = [], [], []
        X.append(wv[val][w2i[val][int_word]])
        wname.append(int_word)
        colors.append('green')
        for word in sorted(total_neighbors):
          if word in w2i[val]:
            X.append(wv[val][w2i[val][word]])
            wname.append(word)
            colors.append(neighbor2color[word])
        X = np.array(X, dtype=np.float)
        embeddings = TSNE(n_components=2, verbose=2, perplexity=30, n_iter=1000).fit_transform(X)
        xx, yy = embeddings[:, 0], embeddings[:, 1]
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.scatter(xx, yy, c=colors)
        plt.title('t-SNE for word %s in space %s'%(int_word, val.split('_')[0]))
        for wi, word in enumerate(wname):
          if wi == 0:
            plt.annotate(word, xy=(xx[wi], yy[wi]), xytext=(xx[wi], yy[wi]), textcoords="data", fontsize=20)
          if wi%1==0:
            plt.annotate(word, xy=(xx[wi], yy[wi]), xytext=(xx[wi], yy[wi]), textcoords="data", fontsize=10)
        #plt.show()
        fig.savefig(args.out_dir+"/%s_%s_%s_sp%s_w%s.pdf"%(property, val1.split('_')[0], val2, val.split('_')[0], int_word), bbox_inches='tight')
    else:
      print('skipping word %s'%int_word)

if __name__ == '__main__':
  args = parser.parse_args()
  values = {  'occupation': ('performer', 'sports'),
              'birthyear': ('1990_2009', '1950_1969'),
              'gender': ('male', 'female'),
              'fame': ('rising', 'superstar'),
              'hebrew': ('2014', '2018'),
              'french': ('2014', '2018'),
              'yang': ('weekday', 'weekend')
           }
  property = args.property
  if property == 'occupation2':
    property = 'occupation'
    val1, val2 = 'creator', 'sports'
  elif property == 'occupation3':
    property = 'occupation'
    val1, val2 = 'creator', 'performer'
  else:
    val1, val2 = values[property]
  vocab, wv, w2i = load_all_embeddings(property, val1, val2)
  tsne_plot(property, val1+"_a", val2)



