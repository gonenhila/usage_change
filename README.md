## Detecting Words with Usage Change across Corpora

Code used in our EMNLP'19 submission titled 'Simple, Interpretable and Stable Method for Detecting Words with Usage Change across Corpora'.

### Intro
The problem of comparing corpora and searching for words that differ in their usage arises often in computational social science. This is commonly approached by aligning the word embeddings of each corpus, and looking for words whose cosine distance in the aligned space is large. However, these methods often require extensive filtering and are unstable. We propose an alternative approach that does not use vector space alignment, and instead considers the neighbors of each word. The method is simple, interpretable and stable. We demonstrate its effectiveness on 9 different corpus splitting criteria (age, gender and profession of tweet authors, time of tweet) and different languages (English, French and Hebrew).

t-SNE visualization of top-50 neighbors from each corpus for word `clutch`, Gender split.
![Word `clutch` in gender split](images/gender.png)

t-SNE visualization of top-50 neighbors from each corpus for word `dam`, Age split.
![Word `dam` in age split](images/age.png)


### Dependencies
* nltk
* numpy
* scipy
* sklearn
* matplotlib
* gensim (for custom datasets)

### Quick Start

#### Use standard subcorpus employed in our paper
* Download the data, embeddings, vocab files used in our paper from [Drive](https://drive.google.com/open?id=1ytwtPNZGs7DfoLavsfkw5DaIa-PkhACG). Untar tokdata.tar.gz to tokdata/
* Open sources/run.sh, set the variables `STD_DIR` to the path of the extracted data and and `RES_DIR` to the path where our method outputs and visualization plots will be stored.
* Run our detection method on Age split (`young` vs. `old`):
```
bash sources/run.sh detect standard young old
```
will save the most changed words, stability scores and so on at `$RES_DIR/detect_young_old_*`.
Other splits include: `male`, `female`, `performer`, `creator`, `sports`, `weekday`, `weekend`, `hebrew2014`, `hebrew2018`, `french2014` and `french2018`.
* Visualize the nearest neighbors of given words (`dam`, `assist`) in two subspaces:
```
bash sources/run.sh visualize standard young old dam,assist
```
will save the two plots for each word as pdf at `$RES_DIR/vis_young_old*`.

#### Use custom subcorpus
* Ensure you preprocess the subcorpus (e.g., tokenization, handling numbers, removing URLs) and keep each sentence in a single line.
* Open sources/run.sh, set the variables `STD_DIR` to the path of the extracted data and and `RES_DIR` to the path where our method outputs and visualization plots will be stored.
* Train word2vec for both the subcorpus (say `fake_posts` and `real_posts`):
```
bash sources/run.sh train custom fake_posts fake real_posts real
```
will save the embeddings, vocab files at `$RES_DIR/fake*` and `$RES_DIR/real*`. Note that `fake` and `real` are shorthand names for the two subcorpus, mainly for easy bookkeeping.
* Run our detection method on the custom split:
```
bash sources/run.sh detect custom fake_posts real_posts fake real
```
will save the most changed words, stability scores and so on at `$RES_DIR/detect_fake_real_*`.
* Visualize the nearest neighbors of given words (`news`, `truth`) in two subspaces:
```
bash sources/run.sh visualize custom fake_posts real_posts fake real news,truth
```
will save the two plots for each word as pdf at `$RES_DIR/vis_fake_real_*`.

### License
This repository is GPL-licensed.
