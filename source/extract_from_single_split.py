# word identification code

import numpy as np
import pickle
from scipy.spatial.distance import cosine
import scipy
from numpy import linalg as LA
import random
from tqdm import tqdm
import sys
import argparse
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import operator
import sys
import pdb

random.seed(10)

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
parser.add_argument("--out_topk", 
                    default='/tmp/result', 
                    help="prefix for the output files for topk and latex table")
parser.add_argument("--freq_thr", type=float, default=0.00001, help="frequency threshold")
parser.add_argument("--min_count", type=int, default=200, help="min appearances of a word")
parser.add_argument("--k", type=int, default=1000, help="k of k-NN to use")

def load_embeddings_hamilton(filename):
    # loads word embeddings and vocab file from hamilton codebase

    # load vocab file
    print('loading ...')
    vocab_file = open(filename + '-vocab.pkl', 'r')
    data = pickle.load(vocab_file)
    vocab = [line.strip() for line in data]
    w2i = {w: i for i, w in enumerate(vocab)}

    # load word embeddings
    wv = np.load(filename + '-w.npy')

    return vocab, wv, w2i

def load_embeddings_from_np(filename):
    # load word embedding file and vocab file from file_prefix
    print('loading %s'%filename)
    with open(filename + '.vocab', 'r') as f_embed:
        vocab = [line.strip() for line in f_embed]
    w2i = {w: i for i, w in enumerate(vocab)}
    wv = np.load(filename + '.wv.npy')
    return vocab, wv, w2i

def normalize(wv):
    # normalize vectors
    norms = np.apply_along_axis(LA.norm, 1, wv)
    wv = wv / norms[:, np.newaxis]
    return wv

def load_and_normalize(lang, filename, vocab, wv, w2i, hamilton=False):
    # load word embeddings, vocab file and update the global maps (vocab, wv, w2i)

    # load word embeddings, vocab file 
    if hamilton:
        vocab_muse, wv_muse, w2i_muse = load_embeddings_hamilton(filename)
    else:
        vocab_muse, wv_muse, w2i_muse = load_embeddings_from_np(filename)

    # normalize the word embeddings
    wv_muse = normalize(wv_muse)

    # update the global maps
    vocab[lang] = vocab_muse
    wv[lang] = wv_muse
    w2i[lang] = w2i_muse
    print('loaded and normalized %s embeddings'%filename)

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


def similarity(w1, w2, space):
    # dot product of vectors corresponding to two words (w1 and w2) in a space
    i1 = w2i[space][w1]
    i2 = w2i[space][w2]
    vec1 = wv[space][i1, :]
    vec2 = wv[space][i2, :]
    return np.inner(vec1, vec2)

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

    # top-frequent
    top_freq = defaultdict(int)
    sorted_words = [x[0] for x in sorted(count_vocab.items(), key=operator.itemgetter(1))]
    cutoff = len(sorted_words) / float(20)
    top_freq_words = sorted_words[int(4 * cutoff):-200]  # -int(cutoff)]
    for w in top_freq_words:
        top_freq[w] = count[w]

    print('done')
    return freq_norm, count_vocab, top_freq

def load_all_embeddings(args):
    # loads embedding from both spaces and both seeds (123, 456) and perform alignment
    vocab = {}
    wv = {}
    w2i = {}
    load_and_normalize(val1+'0', args.embed_a, vocab, wv, w2i)
    load_and_normalize(val2+'0', args.embed_b, vocab, wv, w2i)
    load_and_normalize(val1+'1', args.embed_a.replace('seed123', 'seed456'), vocab, wv, w2i)
    load_and_normalize(val2+'1', args.embed_b.replace('seed123', 'seed456'), vocab, wv, w2i)
    align(w2i, wv, vocab, val1+'0', val2+'0', val1+'_a0')
    align(w2i, wv, vocab, val1+'1', val2+'1', val1+'_a1')
    return vocab, wv, w2i

def cosdist_scores(space1, space2, freq1, freq2, count1, count2):
    # compute cosine similarity for common words and sort these words from low similarity to high

    # compute cosine similarity for common words
    all_scores = []
    pbar = tqdm(total=len(vocab[space1]))
    for i, w in enumerate(vocab[space1]):
        if w not in s_words and w in freq1 and w in freq2 and count1[w] > MIN_COUNT and count2[w] > MIN_COUNT:
            all_scores.append((np.inner(wv[space1][w2i[space1][w], :], wv[space2][w2i[space2][w], :]), w))
        if i%10 == 0:
            pbar.update(10)
    pbar.close()
    print('len of ranking', len(all_scores))

    # rank these words
    all_scores_sorted = sorted(all_scores)
    return all_scores_sorted


def NN_scores(space1, sapce2, freq1, freq2, count1, count2):
    # rank all the words based on our method  
    
    # compute nearest neighbors overlap for all the common words
    nn_scores = []
    pbar = tqdm(total=len(vocab[space1]))
    for i, w in enumerate(vocab[space1]):
        if w not in s_words and w in freq1 and w in freq2 and count1[w] > MIN_COUNT and count2[w] > MIN_COUNT:
            neighbors_bef = set(topK(w, space1, args.k, count1, 100))
            neighbors_aft = set(topK(w, sapce2, args.k, count2, 100))
            nn_scores.append((len(neighbors_bef.intersection(neighbors_aft)), w))
        if i%10 == 0:
            pbar.update(10)
    pbar.close()
    print('len of ranking', len(nn_scores))

    # rank these words
    nn_scores_sorted = sorted(nn_scores)
    return nn_scores_sorted


def correlation(cosdist, nn):
    # compute correlation of rankings for different seeds
    nn_1 = sorted(nn[0], key=lambda x: x[1])
    nn_2 = sorted(nn[1], key=lambda x: x[1])
    cosdist_1 = sorted(cosdist[0], key=lambda x: x[1])
    cosdist_2 = sorted(cosdist[1], key=lambda x: x[1])
    nn_scores_1 = [item[0] + random.random() for item in nn_1]
    nn_scores_2 = [item[0] + random.random() for item in nn_2]
    cosdist_scores_1 = [item[0] for item in cosdist_1]
    cosdist_scores_2 = [item[0] for item in cosdist_2]
    return spearmanr(cosdist_scores_1, cosdist_scores_2)[0], spearmanr(nn_scores_1, nn_scores_2)[0]


def precision_at_k(cosdist, nn):
    # compute precision at k of rankings for different seeds

    # prepare the rankings for different seeds
    nn_1 = [item[1] for item in nn[0]]
    cosdist_1 = [item[1] for item in cosdist[0]]
    nn_2 = [item[1] for item in nn[1]]
    cosdist_2 = [item[1] for item in cosdist[1]]

    # calculate precision@k
    all_nn = []
    all_cosdist = []
    for k in [10, 20, 50, 100, 200, 500, 1000]:
        nn_p = len([item for item in nn_1[:k] if item in nn_2[:k]]) / float(k)
        cosdist_p = len([item for item in cosdist_1[:k] if item in cosdist_2[:k]]) / float(k)
        all_nn.append(nn_p)
        all_cosdist.append(cosdist_p)
    
    print('precision at k for cosdist', all_cosdist)
    print('precision at k for nn', all_nn)
    return all_cosdist, all_nn

def diff_nn(w, count1, count2, MIN_COUNT):
    # identify top 10 neighbors that are specific in a space

    # identify the neighbors 
    nn1 = topK(w, val1+'0', 1000, count1, 100)
    nn2 = topK(w, val2+'0', 1000, count2, 100)

    # identify the top 10 neighbors that specific to a space
    top_diff1 = []
    top_diff2 = []
    for w in nn1:
        if w not in nn2:
            top_diff1.append(w)
            if len(top_diff1) == 10:
                break
    for w in nn2:
        if w not in nn1:
            top_diff2.append(w)
            if len(top_diff2) == 10:
                break
    return top_diff1, top_diff2

def print_to_file(filename, precisions_cosdist, precisions_nn, cosdist, nn, corr_cosdist, corr_nn, count_vocab_val1, count_vocab_val2, k =10):
    # print results to a file

    var_dict = {'cosdist': cosdist, 'nn': nn}
    with open(filename+'.txt', 'w') as f, open(filename+'_latex.txt', 'w') as f_latex:
        #f.write('\n' + property + '\n=*=*=*=*=*=*=\n')
        #f_latex.write('\n' + property + '\n=*=*=*=*=*=*=\n')
        assert(len(cosdist[0]) == len(nn[0]))
        f.write('length of vocabularies: {} {}, length of rankings: {}\n'.format(len(vocab[val1+'0']), len(vocab[val1+'1']), len(nn[0])))
        for method in ['cosdist', 'nn']:
            f.write('\n' + method + ' top10\n=================\n')
            for w in var_dict[method][0][:k]:
                #top_diff1, top_diff2 = diff_nn(w[1])
                #f.write('{}\n{}\n{}\n'.format(w[1], ', '.join(top_diff1), ', '.join(top_diff2)))
                f.write('{}\n'.format(w[1]))
        f.write('\nprecisions_cosdist\n{}\n'.format(precisions_cosdist))
        f.write('\nprecisions_nn\n{}\n'.format(precisions_nn))
        f.write('\ncorrelation_cosdist\n{}\n'.format(corr_cosdist))
        f.write('\ncorrelation_nn\n{}\n'.format(corr_nn))

        for method in ['cosdist', 'nn']:
            f.write('\n' + method + ' top 100 explained\n=================\n')
            for w in var_dict[method][0][:100]:
                top_diff1, top_diff2 = diff_nn(w[1], count_vocab_val1, count_vocab_val2, MIN_COUNT)
                f.write('\n{}\ncount in corpus1: {}, count in corpus2: {}, measure: {}\n{}\n{}\n'.format(
                    w[1], count_vocab_val1[w[1]], count_vocab_val2[w[1]], w[0], ', '.join(top_diff1), ', '.join(top_diff2)))

        for w_cosdist, w_nn in zip(cosdist[0][:k], nn[0][:k]):
            top_diff1, top_diff2 = diff_nn(w_nn[1], count_vocab_val1, count_vocab_val2, MIN_COUNT)
            f_latex.write('\\multirow{{2}}{{*}} {{{}}} & \\multirow{{2}}{{*}}{{{}}} &  {} \\\\   \
                & &  {}\\\\ \\hline\n'.format(w_cosdist[1], w_nn[1], ', '.join(top_diff1), ', '.join(top_diff2)))


def detect(args):
    # run the word identification method, perform stability experiments and print the results

    # extract frequencies
    freq_norm_val1, count_vocab_val1, top_freq_val1 = extract_freqs(
        args.data_a, vocab[val1 + '0'])
    freq_norm_val2, count_vocab_val2, top_freq_val2 = extract_freqs(
        args.data_b, vocab[val2 + '0'])

    # detect words using cosdist
    print('detecting words using cosdist ...')
    cosdist = []
    for i in range(2):
        cosdist.append(
            cosdist_scores(val1 + '_a' + str(i), val2 + str(i), top_freq_val1, top_freq_val2, count_vocab_val1,
                           count_vocab_val2))
    print('done.')

    # detect words using nn
    print('detecting words using NN ...')
    nn = []
    for i in range(2):
        nn.append(NN_scores(val1 + '_a' + str(i), val2 + str(i), top_freq_val1, top_freq_val2, count_vocab_val1,
                            count_vocab_val2))
    print('done.')

    # stability experiments
    corr_cosdist, corr_nn = correlation(cosdist, nn)
    print('correlation for cosdist {}\ncorrelation for nn {}\n'.format(corr_cosdist, corr_nn))
    precisions_cosdist, precisions_nn = precision_at_k(cosdist, nn)

    print_to_file(args.out_topk, precisions_cosdist, precisions_nn, cosdist, nn, corr_cosdist, corr_nn, count_vocab_val1, count_vocab_val2)


if __name__ == '__main__':
    assert(sys.version_info[0] > 2)
    args = parser.parse_args()
    print(args)
    MIN_COUNT = args.min_count

    s_words = set(stopwords.words('english'))
    val1, val2 = args.name_split_a, args.name_split_b
    vocab, wv, w2i = load_all_embeddings(args)
    detect(args)


