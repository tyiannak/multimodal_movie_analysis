# textual - based feature extraction

### Introductions

Scripts/functionalities to be used for feature extraction using the subtitles of movies.
 
At the end of this process each movie has a dedicated feature vector extracted.

Initial work using:

- Tf-idf Model (tfidf) DONE!
- Non-Negative Matrix Factorization Model (nmf) DONE!
- Latent Dirichlet Allocation Model (lda) DONE!
- Latent Semantic Indexing Model (lsi) TODO!
- Shallow Embedding Model with word2vec, glove, fastText (word2vec) TODO!

Specific details on data paths, models to be used, specs of the models etc. will be described through the corresponding **settings file**. 

An example settings file is provided, namely **settings.yaml**.


### Command:

```python
python analyze_text
```

