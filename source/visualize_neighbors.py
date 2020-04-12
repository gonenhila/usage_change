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
parser.add_argument("--embed_a", 
                    default='usage_change/embeddings/birthyear.1950_1969.lowercase.seed123.mfreq20', 
                    help="prefix for embedding and vocab file for split a")
parser.add_argument("--embed_b", 
                    default='usage_change/embeddings/birthyear.1990_2009.lowercase.seed123.mfreq20', 
                    help="prefix for embedding and vocab file for split b")
parser.add_argument("--data_a", 
                    default='usage_change/tokdata/birthyear.1950_1969.lowercase', 
                    help="name of tokenized data file for split a")
parser.add_argument("--data_b", 
                    default='usage_change/tokdata/birthyear.1990_2009.lowercase', 
                    help="name of tokenized data file for split b")
parser.add_argument("--name_split_a", 
                    default='old', 
                    help="short name for split a")
parser.add_argument("--name_split_b", 
                    default='young', 
                    help="short name for split b")
parser.add_argument("--words", default='pearl', help="interesting words to plot in csv format")
parser.add_argument("--out_dir", default='/tmp/plot_', help="prefix for directory to save the plots")
parser.add_argument("--k", type=int, default=50, help="k of k-NN to use")

def extract_freqs(filename, vocab):
  # extract word raw frequencies and normalized frequencies

  # raw counts
  print('extracting freqs %s'%filename)
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
  # load word embedding file and vocab file from file_prefix
  print('loading %s'%filename)
  with open(filename + '.vocab', 'r') as f_embed:
    vocab = [line.strip() for line in f_embed]
  w2i = {w: i for i, w in enumerate(vocab)}
  wv = np.load(filename + '.wv.npy')
  return vocab, wv, w2i

def load_and_normalize(idi, filename, vocab, wv, w2i, hamilton=False):
  # load word embeddings, vocab file and update the global maps (vocab, wv, w2i)

  # load word embeddings, vocab file
  cur_vocab, cur_wv, cur_w2i = load_embeddings_from_np(filename)

  # normalize the word embeddings
  cur_wv = normalize(cur_wv)

  # update the global maps
  vocab[idi] = cur_vocab
  wv[idi] = cur_wv
  w2i[idi] = cur_w2i
  print('loaded and normalized %s embeddings'%filename)

def topK(w, space, k=10, count = None, min_freq = 0):
  # identify the top k neighbors of a word in a space

  # extract the word vector for word w
  idx = w2i[space][w]
  vec = wv[space][idx, :]

  # compute similarity of w with all words in the vocabulary
  sim = wv[space].dot(vec)

  # sort similarities by descending order
  sort_sim = (sim.argsort())[::-1]

  # choose topK
  if count:
    # consider only the neighbors whose raw frequency is greater than min_freq
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
  # update the global maps with the aligned vectors, vocab and word to sequence id mapping
  wv[aligned] = np.zeros(wv[space1].shape)
  for i, vec in enumerate(wv[space1]):
    wv[aligned][i, :] = np.dot(vec, Q_bef)
  vocab[aligned] = vocab[space1]
  w2i[aligned] = w2i[space1]


def align(w2i, wv, vocab, space1, space2, aligned):
  # align the word embeddings from two spaces using orthogonal procrustes (OP)

  # identify the common words in both spaces
  train_words = list(set(vocab[space1]).intersection(set(vocab[space2])))
  
  # perform OP
  num = len(train_words)
  mat_bef = np.zeros((num, 300))
  mat_aft = np.zeros((num, 300))
  for i, w in enumerate(train_words):
    mat_bef[i, :] = wv[space1][w2i[space1][w]]
    mat_aft[i, :] = wv[space2][w2i[space2][w]]
  Q_bef, s_bef = scipy.linalg.orthogonal_procrustes(mat_bef, mat_aft)
  
  # update the global maps
  create_aligned(w2i, wv, vocab, Q_bef, space1, aligned)

  # normalize the aligned embeddings
  wv[aligned] = normalize(wv[aligned])

def load_all_embeddings(args):
  # update the global maps with embeddings, w2i, i2w maps for both corpus
  vocab = {}
  wv = {}
  w2i = {}
  load_and_normalize(val1, args.embed_a, vocab, wv, w2i)
  load_and_normalize(val2, args.embed_b, vocab, wv, w2i)
  align(w2i, wv, vocab, val1, val2, val1+'_a')
  return vocab, wv, w2i

def tsne_plot(property, val1, val2):
  # visualize the top-k neighbors of each interesting word in both spaces

  # extract frequencies
  freq_norm_val1, count_vocab_val1 = extract_freqs(
      args.data_a, vocab[val1])
  freq_norm_val2, count_vocab_val2 = extract_freqs(
      args.data_b, vocab[val2])
  
  # run through all the words of interest
  for int_word in args.words.strip().split(","):
    # ensure the word is in both spaces
    if int_word in vocab[val1] and int_word in vocab[val2]:
      # identify the top-k neighbors
      neighbors_a = set(topK(int_word, val1, args.k, None, 100))
      neighbors_b = set(topK(int_word, val2, args.k, None, 100))
      total_neighbors = neighbors_a.union(neighbors_b)
      # identify neighbors which occur in a specific space and common space
      neighbor2color = {int_word:'green'} # coloring code - green for word of interest
      common, na, nb = [], [], [] # na, nb contains neighbors which are in specific space
      for neighbor in total_neighbors:
        if neighbor in neighbors_a and neighbor in neighbors_b:
          neighbor2color[neighbor] = 'purple' # coloring code - purple for neighbors in common space
          common.append(neighbor)
        elif neighbor in neighbors_a:
          neighbor2color[neighbor] = 'cyan' # coloring code - cyan for neighbors in space 'a'
          na.append(neighbor)
        else:
          neighbor2color[neighbor] = 'violet' # coloring code - violet for neighbors in space 'b'
          nb.append(neighbor)
      
      # run over each space
      for val in [val1, val2]:
        # construct embedding matrix (tsne input) for neighboring words
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
        # perform tsne
        embeddings = TSNE(n_components=2, verbose=2, perplexity=30, n_iter=1000).fit_transform(X)
        # make tsne plot
        xx, yy = embeddings[:, 0], embeddings[:, 1]
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.scatter(xx, yy, c=colors)
        plt.title('t-SNE for word %s in space %s'%(int_word, val))
        for wi, word in enumerate(wname):
          if wi == 0:
            plt.annotate(word, xy=(xx[wi], yy[wi]), xytext=(xx[wi], yy[wi]), textcoords="data", fontsize=20)
          if wi%1==0:
            plt.annotate(word, xy=(xx[wi], yy[wi]), xytext=(xx[wi], yy[wi]), textcoords="data", fontsize=10)
        fig.savefig(args.out_dir+"sp%s_w%s.pdf"%(val, int_word), bbox_inches='tight')
    else:
      print('skipping word %s'%int_word)

if __name__ == '__main__':
  args = parser.parse_args()
  val1, val2 = args.name_split_a, args.name_split_b
  vocab, wv, w2i = load_all_embeddings(args)
  tsne_plot(args, val1+"_a", val2)



