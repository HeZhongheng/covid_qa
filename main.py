import os
import json
import nltk
import gzip
import string

from nltk.corpus import stopwords
from pyhanlp import *
from transformers import AutoTokenizer, T5ForConditionalGeneration


DATA_PATH = 'data/'
ClusterAnalyzer = JClass('com.hankcs.hanlp.mining.cluster.ClusterAnalyzer')


def load_context(data_path):
    # load context from tweet files
    contexts = []
    files = os.listdir(data_path)
    for file_name in files:
        file_path = os.path.join(data_path, file_name)
        with gzip.open(file_path) as f:
            for line in f:
                content = json.loads(line)
                # use full text as content
                if 'full_text' in content:
                    contexts.append(content['full_text'])
    return contexts


def load_model():
    # load model
    model_name = "allenai/unifiedqa-t5-small"  # you can specify the model size here
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


def load_stopwords():
    # use stopwords from nltk
    _stopwords = stopwords.words('english')
    # add punctuation
    for char in string.punctuation:
        _stopwords.append(char)
    return _stopwords


def cluster_answers(answers):
    # cluster answers using repeated bisection clustering
    analyzer = ClusterAnalyzer()
    for answer in answers:
        analyzer.addDocument(answer, answer)
    clusters = analyzer.repeatedBisection(1.0)
    # choose the shortest one as cluster label
    clustered_answers = []
    for cluster in clusters:
        answer = sorted(cluster, key=lambda x:len(x))
        clustered_answers.append(answer)
    return clustered_answers


def main(input_string):
    tokenizer, model = load_model()
    print('model loaded')

    contexts = load_context(DATA_PATH)
    print('context loaded')

    # load stopwords
    _stopwords = load_stopwords()

    # get question keywords by removing stop words from question
    q_keywords = set([word for word in nltk.word_tokenize(input_string.lower())
                     if word not in _stopwords])

    answers = []

    for context in contexts:

        # get context keywords
        context_keywords = set([word for word in nltk.word_tokenize(context.lower())
                                if word not in stopwords.words('english')])

        # using keywords to filter some of context
        intersection = q_keywords & context_keywords
        if len(intersection) < 2: continue

        # get input and inference
        input_string = input_string.lower() + '\n' + context.lower()

        # filter length > 512
        if len(input_string) > 512: continue
        input_ids = tokenizer.encode(input_string, return_tensors="pt")
        res = model.generate(input_ids)
        output_string = tokenizer.batch_decode(res, skip_special_tokens=True)

        answers.extend(output_string)

    answers = cluster_answers(answers)
    for answer in answers:
        print(answer)


if __name__ == '__main__':
    main('What is the origin of COVID-19?')


