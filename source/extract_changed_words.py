
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
random.seed(10)


parser = argparse.ArgumentParser()

parser.add_argument("--out_topk", default='../data/results/topk.txt', help="name of output file for topk")
parser.add_argument("--out_stability", default='../data/results/topk.txt', help="name of output file for stability")
parser.add_argument("--freq_thr", default=0.00001, help="frequency threshold")
parser.add_argument("--k", default=1000, help="k of k-NN to use")


def load_embeddings_hamilton(filename):
    print('loading ...')
    vocab_file = open(filename + '-vocab.pkl', 'r')
    data = pickle.load(vocab_file)
    vocab = [line.strip() for line in data]

    w2i = {w: i for i, w in enumerate(vocab)}
    wv = np.load(filename + '-w.npy')

    return vocab, wv, w2i


def load_embeddings_from_np(filename):
    print('loading ...')
    #with codecs.open(filename + '.vocab', 'r', 'utf-8') as f_embed:
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
    if hamilton:
        vocab_muse, wv_muse, w2i_muse = load_embeddings_hamilton(filename)
    else:
        vocab_muse, wv_muse, w2i_muse = load_embeddings_from_np(filename)
    wv_muse = normalize(wv_muse)
    vocab[lang] = vocab_muse
    wv[lang] = wv_muse
    w2i[lang] = w2i_muse
    print('done')


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


def topK(w, space, k=10):
    # extract the word vector for word w
    idx = w2i[space][w]
    vec = wv[space][idx, :]

    # compute similarity of w with all words in the vocabulary
    sim = wv[space].dot(vec)
    # sort similarities by descending order
    sort_sim = (sim.argsort())[::-1]

    # choose topK
    best = sort_sim[:(k + 1)]

    return [vocab[space][i] for i in best if i != idx]


def similarity(w1, w2, space):
    i1 = w2i[space][w1]
    i2 = w2i[space][w2]
    vec1 = wv[space][i1, :]
    vec2 = wv[space][i2, :]

    return np.inner(vec1, vec2)


def extract_freqs(filename, vocab):
    # raw counts
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
    return freq_norm, count_vocab


def load_all_embeddings():

    vocab = {}
    wv = {}
    w2i = {}

    load_and_normalize('per0', '../data/stability/occupation/run1/occupation.performer.lowercase.seed123.mfreq20', vocab, wv, w2i)
    load_and_normalize('sport0', '../data/stability/occupation/run1/occupation.sports.lowercase.seed123.mfreq20', vocab, wv, w2i)
    load_and_normalize('per1', '../data/stability/occupation/run2/occupation.performer.lowercase.seed456.mfreq20', vocab, wv, w2i)
    load_and_normalize('sport1', '../data/stability/occupation/run2/occupation.sports.lowercase.seed456.mfreq20', vocab, wv, w2i)
    align(w2i, wv, vocab, 'per0', 'sport0', 'per_a0')
    align(w2i, wv, vocab, 'per1', 'sport1', 'per_a1')

    load_and_normalize('old0', '../data/stability/birthyear/run1/birthyear.1950_1969.lowercase.seed123.mfreq20', vocab, wv, w2i)
    load_and_normalize('young0', '../data/stability/birthyear/run1/birthyear.1990_2009.lowercase.seed123.mfreq20', vocab, wv, w2i)
    load_and_normalize('old1', '../data/stability/birthyear/run2/birthyear.1950_1969.lowercase.seed456.mfreq20', vocab, wv, w2i)
    load_and_normalize('young1', '../data/stability/birthyear/run2/birthyear.1990_2009.lowercase.seed456.mfreq20', vocab, wv, w2i)
    align(w2i, wv, vocab, 'old0', 'young0', 'old_a0')
    align(w2i, wv, vocab, 'old1', 'young1', 'old_a1')

    load_and_normalize('male0', '../data/stability/gender/run1/gender.male.lowercase.seed123.mfreq20', vocab, wv, w2i)
    load_and_normalize('female0', '../data/stability/gender/run1/gender.female.lowercase.seed123.mfreq20', vocab, wv, w2i)
    load_and_normalize('male1', '../data/stability/gender/run2/gender.male.lowercase.seed456.mfreq20', vocab, wv, w2i)
    load_and_normalize('female1', '../data/stability/gender/run2/gender.female.lowercase.seed456.mfreq20', vocab, wv, w2i)
    align(w2i, wv, vocab, 'male0', 'female0', 'male_a0')
    align(w2i, wv, vocab, 'male1', 'female1', 'male_a1')

    return vocab, wv, w2i


def cosdist_scores(space1, space2, freq1, freq2):
    all_scores = []
    print(len(vocab[space1]))
    for i, w in tqdm(enumerate(vocab[space1])):
        assert (w in freq1)
        if w not in s_words and w in freq2 and freq1[w] > MIN_FREQ and freq2[w] > MIN_FREQ:
            all_scores.append((np.inner(wv[space1][w2i[space1][w], :], wv[space2][w2i[space2][w], :]), w))
        if i>150:
            break
    all_scores_sorted = sorted(all_scores)
    return all_scores_sorted


def NN_scores(space1, sapce2, freq1, freq2):
    nn_scores = []
    for i, w in tqdm(enumerate(vocab[space1])):
        assert (w in freq1)
        if w not in s_words and w in freq2 and freq1[w] > MIN_FREQ and freq2[w] > MIN_FREQ:
            neighbors_bef = set(topK(w, space1, args.k))
            neighbors_aft = set(topK(w, sapce2, args.k))
            nn_scores.append((len(neighbors_bef.intersection(neighbors_aft)), w))
        if i>150:
            break
    nn_scores_sorted = sorted(nn_scores)
    return nn_scores_sorted


def correlation(cosdist, nn):
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
    nn_1 = [item[1] for item in nn[0]]
    cosdist_1 = [item[1] for item in cosdist[0]]

    nn_2 = [item[1] for item in nn[1]]
    cosdist_2 = [item[1] for item in cosdist[1]]

    all_nn = []
    all_cosdist = []

    for k in [10, 20, 50, 100, 200, 500, 1000]:
        nn_p = len([item for item in nn_1[:k] if item in nn_2[:k]]) / float(k)
        cosdist_p = len([item for item in cosdist_1[:k] if item in cosdist_2[:k]]) / float(k)
        all_nn.append(nn_p)
        all_cosdist.append(cosdist_p)

    print('precision at k for cosdist', all_cosdist)
    print('precision at k for nn', all_nn)
    return all_nn, all_cosdist


def print_topk_to_file(filename, k =10):

    var_dict = {'cosdistocc': cosdist_occ, 'cosdistgender': cosdist_gender, 'cosdistage': cosdist_age,
           'nnocc': nn_occ, 'nngender': nn_gender, 'nnage': nn_age,}
    with open(filename, 'w') as f:
        for split in ['occ', 'age', 'gender']:
            f.write('\n' + split + '\n=*=*=*=*=*=*=\n')
            for method in ['cosdist', 'nn']:
                f.write(method + '\n=========\n')
                for w in var_dict[method+split][0][:k]:
                    f.write(w[1]+'\n')

    return



if __name__ == '__main__':

    args = parser.parse_args()

    vocab, wv, w2i = load_all_embeddings()

    s_words = set(stopwords.words('english'))
    MIN_FREQ = args.freq_thr

    # extract frequencies
    freq_sport, count_sport = extract_freqs('../data/stability/freqs/occupation.sports.lowercase', vocab['sport0'])
    freq_per, count_per = extract_freqs('../data/stability/freqs/occupation.performer.lowercase', vocab['per0'])

    freq_old, count_old = extract_freqs('../data/stability/freqs/birthyear.1950_1969.lowercase', vocab['old0'])
    freq_young, count_young = extract_freqs('../data/stability/freqs/birthyear.1990_2009.lowercase', vocab['young0'])

    freq_male, count_male = extract_freqs('../data/stability/freqs/gender.male.lowercase', vocab['male0'])
    freq_female, count_female = extract_freqs('../data/stability/freqs/gender.female.lowercase', vocab['female0'])


    # detect words using cosdist
    print('detect words using cosdist ...')
    cosdist_occ = []
    cosdist_gender = []
    cosdist_age = []

    for i in range(2):
        cosdist_occ.append(cosdist_scores('per_a'+str(i), 'sport'+str(i), freq_per, freq_sport))
        cosdist_gender.append(cosdist_scores('male_a'+str(i), 'female'+str(i), freq_male, freq_female))
        cosdist_age.append(cosdist_scores('old_a'+str(i), 'young'+str(i), freq_old, freq_young))
    print('done.')

    # detect words using nn
    print('detect words using NN ...')
    nn_occ = []
    nn_gender = []
    nn_age = []

    for i in range(2):
        nn_occ.append(NN_scores('per_a'+str(i), 'sport'+str(i), freq_per, freq_sport))
        nn_gender.append(NN_scores('male_a'+str(i), 'female'+str(i), freq_male, freq_female))
        nn_age.append(NN_scores('old_a'+str(i), 'young'+str(i), freq_old, freq_young))
    print('done.')

    # stability experiments
    correlation(cosdist_occ, nn_occ)
    correlation(cosdist_age, nn_age)
    correlation(cosdist_gender, nn_gender)

    all_nn, all_cosdist = precision_at_k(cosdist_occ, nn_occ)
    all_nn, all_cosdist = precision_at_k(cosdist_age, nn_age)
    all_nn, all_cosdist = precision_at_k(cosdist_gender, nn_gender)

    print_topk_to_file(args.out_topk)