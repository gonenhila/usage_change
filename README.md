## Detecting Words with Usage Change across Corpora

Code used in our ACL 2020 paper titled 'Simple, Interpretable and Stable Method for Detecting Words with Usage Change across Corpora'.

### Intro
The problem of comparing two bodies of text and searching for words that differ in their usage between them arises often in digital humanities and computational social science. This is commonly approached by training word embeddings on each corpus, aligning the vector spaces, and looking for words whose cosine distance in the aligned space is large. However, these methods often require extensive filtering of the vocabulary to perform well, and---as we show in this work---result in unstable, and hence less reliable, results. We propose an alternative approach that does not use vector space alignment, and instead considers the neighbors of each word. The method is simple, interpretable and stable. We demonstrate its effectiveness in 9 different setups, considering different corpus splitting criteria (age, gender and profession of tweet authors, time of tweet) and different languages (English, French and Hebrew).

t-SNE visualization of top-50 neighbors from each corpus for word `clutch`, Gender split with cyan for neighbors only in `female` corpus and violet for neighbors only in `male` corpus.
![Word `clutch` in gender split](images/gender.png)

t-SNE visualization of top-50 neighbors from each corpus for word `dam`, Age split with cyan for neighbors only in `older` corpus and violet for neighbors only in `young` corpus.
![Word `dam` in age split](images/age.png)

### Dependencies
* nltk
* numpy
* scipy
* sklearn
* matplotlib
* gensim (for custom datasets)

### Quick Start

#### Use standard splits (e.g. age, gender) studied in our paper
* Download the data, embeddings, vocab files used in our paper from [Drive](https://drive.google.com/open?id=1ytwtPNZGs7DfoLavsfkw5DaIa-PkhACG). Untar tokdata.tar.gz to tokdata/
* Open sources/run.sh, set the variables `STD_DIR` to the path of the extracted data and and `RES_DIR` to the path where our method outputs and visualization plots will be stored.
* Run our detection method on Age split (`young` vs. `old`):
```
bash sources/run.sh detect standard young old
(format: bash sources/run.sh detect standard <split_a> <split_b>)
```
will save the most changed words, stability scores and so on at `$RES_DIR/detect_young_old_*`.
Other splits you can specify include: `male`, `female`, `performer`, `creator`, `sports`, `weekday`, `weekend`, `hebrew2014`, `hebrew2018`, `french2014` and `french2018`.
* Visualize the nearest neighbors of given words (`dam`, `assist`) in two subspaces:
```
bash sources/run.sh visualize standard young old dam,assist
(format: bash sources/run.sh visualize standard <split_a> <split_b> <words_in_csv>)
```
will save the two plots for each word as pdf at `$RES_DIR/vis_young_old*`.

#### Use custom splits
* Ensure you preprocess the subcorpus (e.g., tokenization, handling numbers, removing URLs) and keep each sentence in a single line.
* Open sources/run.sh, set the variables `STD_DIR` to the path of the extracted data and and `RES_DIR` to the path where our method outputs and visualization plots will be stored.
* Train word2vec for both the subcorpus (say `corp1_posts` and `corp2_posts`):
```
bash sources/run.sh train custom corp1_posts corp1 corp2_posts corp2
(format: bash sources/run.sh train custom <custom_split_a_txt_file> <custom_split_a_short_name> <custom_split_b_txt_file> <custom_split_b_short_name>)
```
will save the embeddings, vocab files at `$RES_DIR/corp1*` and `$RES_DIR/corp2*`. Note that `corp1` and `corp2` are shorthand names for the two subcorpus, mainly to aid bookkeeping.
* Run our detection method on the custom split:
```
bash sources/run.sh detect custom corp1_posts corp2_posts corp1 corp2
(format: bash sources/run.sh detect custom <custom_split_a_txt_file> <custom_split_b_txt_file> <custom_split_a_short_name> <custom_split_b_short_name>)
```
will save the most changed words, stability scores and so on at `$RES_DIR/detect_corp1_corp2_*`.
* Visualize the nearest neighbors of given words (`word_a`, `word_b`) in two subspaces:
```
bash sources/run.sh visualize custom corp1_posts corp2_posts corp1 corp2 word_a,word_b
(format: bash sources/run.sh visualize custom <custom_split_a_txt_file> <custom_split_b_txt_file> <custom_split_a_short_name> <custom_split_b_short_name> <words_in_csv>)
```
will save the two plots for each word as pdf at `$RES_DIR/vis_corp1_corp2_*`.

### License
This repository is GPL-licensed.


