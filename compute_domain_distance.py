import argparse
from collections import Counter
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from dataloader import CoNLL2003Reader, i2b22006Reader, i2b22014Reader, \
        i2b22012Reader, EventTSVReader, UnlabeledDataset, CombinedDataset

def read_dataset(identifier, task):
    dataset = None
    if identifier == 'conll2003':
        dataset = CoNLL2003Reader(
                    '../datasets/conll_2003/train.txt',
                    '../datasets/conll_2003/valid.txt',
                    '../datasets/conll_2003/test.txt',
                    task
                ).dataset
    if identifier == 'i2b22006':
        dataset = i2b22006Reader(
                    '../datasets/i2b2_2006/deid_surrogate_train_all_version2.xml',
                    '../datasets/i2b2_2006/deid_surrogate_test_all_groundtruth_version2.xml',
                    task
                ).dataset
    if identifier == 'i2b22010':
        dataset = CoNLL2003Reader(
                    '../datasets/i2b2_2010/train.txt',
                    '',  # i2b2 2010 doesn't have a development set
                    '../datasets/i2b2_2010/test.txt',
                    task
                ).dataset
    if identifier == 'i2b22012':
        dataset = i2b22012Reader(
                    '../datasets/i2b2_2012/full-training-set/',
                    '../datasets/i2b2_2012/ground_truth/merged_i2b2/',
                    task
                ).dataset
    if identifier == 'i2b22014':
        dataset = i2b22014Reader(
                    '../datasets/i2b2_2014/full-training-set/',
                    '../datasets/i2b2_2014/testing-PHI-Gold-fixed/',
                    task
                ).dataset
    if identifier == 'timebank':
        dataset = EventTSVReader('../datasets/hypneg_timebank/', task).dataset
    if identifier == 'mtsamples':
        dataset = EventTSVReader('../datasets/ProcessedRecords/', task).dataset
    return dataset

def renyi_div(probs_p, probs_q, alpha=0.99):
    renyi_div = [(x**alpha)/(y**(alpha-1)) for x,y in zip(probs_p, probs_q) if y!=0.0]
    renyi_div = np.log(sum(renyi_div))
    renyi_div /= (alpha-1)
    return renyi_div

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, action='store', required=True, \
            help='Source dataset [conll2003, i2b22006, i2b22010, i2b22012, i2b22014, timebank, mtsamples]')
    parser.add_argument('--target', type=str, action='store', required=True, \
            help='Target dataset [conll2003, i2b22006, i2b22010, i2b22012, i2b22014, timebank, mtsamples]')
    parser.add_argument('--task', type=str, action='store', required=True, help='Choose task from [ee, ner-fine, ner-coarse]')
    args = parser.parse_args()

    source_dataset = read_dataset(args.source, args.task)
    target_dataset = read_dataset(args.target, args.task)

    source_vocab, target_vocab = Counter(), Counter()
    for example in source_dataset.train_data + source_dataset.dev_data + source_dataset.test_data:
        source_vocab.update(example['tokens'])
    for example in target_dataset.train_data + target_dataset.dev_data + target_dataset.test_data:
        target_vocab.update(example['tokens'])
    
    source_tokens = set(source_vocab.keys())
    target_tokens = set(target_vocab.keys())
    common_tokens = source_tokens.intersection(target_tokens)
    tvo = len(common_tokens) / float(len(target_tokens))
    print('TVO: {}'.format(tvo))

    stop_words = []
    for line in open('stopwords.txt'):
        stop_words.append(line.strip())
    
    punct_list = ['/', '\\', '.', ',', '"', '!', ':', ';', '?', '-', ')', '(', '}', '{', ']', '[', '\'', '...']
    filtered_source_vocab = Counter({k:v for k,v in source_vocab.items() if k not in stop_words and k not in punct_list})
    filtered_target_vocab = Counter({k:v for k,v in target_vocab.items() if k not in stop_words and k not in punct_list})
    merged_vocab = filtered_source_vocab + filtered_target_vocab
    top_words = list(reversed(sorted(merged_vocab.items(), key=lambda x: x[1])))[:10000]
    top_words = [x[0] for x in top_words]

    source_counts = [filtered_source_vocab[x] if x in filtered_source_vocab else 0.0 for x in top_words]
    target_counts = [filtered_target_vocab[x] if x in filtered_source_vocab else 0.0 for x in top_words]
    source_sum = float(sum(source_counts))
    target_sum = float(sum(target_counts))
    source_probs = [x/source_sum for x in source_counts]
    target_probs = [x/target_sum for x in target_counts]
    kldiv = entropy(target_probs, source_probs)
    jsdiv = jensenshannon(target_probs, source_probs)**2
    rendiv = renyi_div(target_probs, source_probs)
    print('KL Divergence: {}'.format(kldiv))
    print('JS Divergence: {}'.format(jsdiv))
    print('Renyi Divergence: {}'.format(rendiv))
