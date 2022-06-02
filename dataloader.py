import random
import os
import re
import math
import xml.etree.ElementTree as ET
import nltk
from nltk import sent_tokenize, word_tokenize
import torch


# Dataloader for CoNLL 2003 and i2b2 NER datasets
class CoNLL2003Reader:

    def __init__(self, train_file, dev_file, test_file, task_string):
        train_examples = self.read_conll_format_file(train_file)
        dev_examples = self.read_conll_format_file(dev_file) if dev_file != '' else []
        test_examples = self.read_conll_format_file(test_file)
        if not dev_examples:
            dev_size = int(math.ceil(0.1 * len(train_examples)))
            dev_examples = train_examples[:dev_size]
            train_examples = train_examples[dev_size:]

        # Dataset construction
        self.dataset = SeqDataset(train_examples, dev_examples, test_examples, task_string)

    def read_conll_format_file(self, filepath):
        reader = open(filepath)
        ex_id = 0
        tokens = []
        labels = []
        examples = []   # Each example is a dict containing ID, tokens and NER labels
        
        for line in reader:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if tokens:
                    examples.append({'id': ex_id, 'tokens': tokens, 'labels': labels})
                    ex_id += 1
                    tokens = []
                    labels = []
            else:
                ex_data = line.split()
                tokens.append(ex_data[0])
                label = ex_data[-1].rstrip()
                labels.append(label if label != 'O' else 'OUT')
        # Last example may be left out
        if tokens:
            examples.append({'id': ex_id, 'tokens': tokens, 'labels': labels})
        return examples


# Dataloader for i2b2 2006 DEID dataset
class i2b22006Reader:

    def __init__(self, train_file, test_file, task_string):
        train_text = self.embed_tags_in_text(train_file)
        test_text = self.embed_tags_in_text(test_file)

        # Example construction
        train_examples = self.construct_examples(train_text)
        test_examples = self.construct_examples(test_text)
        random.shuffle(train_examples)

        dev_size = int(math.ceil(0.1 * len(train_examples)))
        dev_examples = train_examples[:dev_size]
        train_examples = train_examples[dev_size:]

        # Label conversion into common format
        train_examples = self.correct_labels(train_examples)
        dev_examples = self.correct_labels(dev_examples)
        test_examples = self.correct_labels(test_examples)

        # Dataset construction
        self.dataset = SeqDataset(train_examples, dev_examples, test_examples, task_string)

    def embed_tags_in_text(self, file):
        root = ET.parse(open(file))
        doc_text = str(ET.tostring(root.getroot())).replace('\\n', '\n')
        phi_tags = re.findall(re.compile("<PHI [^/]*>[^<]*</PHI>[^\\s]*"), str(doc_text))
        for tag in phi_tags:
            phi_type = tag.split("TYPE=\"")[1].split('"')[0]
            pre_tag_text = tag.split("<")[0]
            post_tag_text = tag.split(">")[-1]
            tag_text = tag.split('</')[0].split('>')[-1].split()
            replacement_tag = pre_tag_text + ' ' + tag_text[0]+'||B-'+phi_type + ' ' + ' '.join([x+'||I-'+phi_type for x in tag_text[1:]]) + ' ' + post_tag_text
            replacement_tag = replacement_tag.strip()
            doc_text = str(doc_text).replace(tag, replacement_tag+' ')
        return doc_text

    def construct_examples(self, tagged_text):
        examples = []
        ex_id = 0
        for line in tagged_text.split('\n'):
            if line.startswith('<ROOT>') or line.startswith('<RECORD') or line.startswith('<TEXT') \
                    or line.startswith('</ROOT>') or line.startswith('</TEXT>') or line.startswith('</RECORD>'):
                continue
            words = line.split()
            tokens = []
            labels = []
            for word in words:
                if '||' not in word:
                    tokens.append(word)
                    labels.append('OUT')
                else:
                    token, label = word.split('||')
                    tokens.append(token)
                    labels.append(label)
            examples.append({'id': ex_id, 'tokens': tokens, 'labels': labels})
            ex_id += 1
        return examples

    def correct_labels(self, examples):
        label_dict = {
            'AGE': 'MISC',
            'ID': 'MISC',
            'PHONE': 'MISC',
            'DOCTOR': 'PER',
            'PATIENT': 'PER',
            'LOCATION': 'LOC',
            'HOSPITAL': 'ORG',
            'DATE': 'MISC'
        }
        for example in examples:
            corrected_labels = []
            for label in example['labels']:
                if '-' not in label:
                    corrected_labels.append(label)
                else:
                    start, tag = label.split('-')
                    corrected_labels.append('{}-{}'.format(start, label_dict[tag]))
            example['labels'] = corrected_labels
        return examples


# Dataloader for i2b2 2014 DEID dataset
class i2b22014Reader:

    def __init__(self, train_folder, test_folder, task_string):
        train_examples = self.read_xml_files(train_folder, 'train')
        test_examples = self.read_xml_files(test_folder, 'test')

        # Example construction
        train_examples = self.construct_examples(train_examples)
        test_examples = self.construct_examples(test_examples)
        random.shuffle(train_examples)

        dev_size = int(math.ceil(0.1 * len(train_examples)))
        dev_examples = train_examples[:dev_size]
        train_examples = train_examples[dev_size:]

        # Label conversion into common format
        train_examples = self.correct_labels(train_examples)
        dev_examples = self.correct_labels(dev_examples)
        test_examples = self.correct_labels(test_examples)

        # Dataset construction
        self.dataset = SeqDataset(train_examples, dev_examples, test_examples, task_string)

    def read_xml_files(self, folder, split):
        texts = []
        for file in os.listdir(folder):
            root = ET.parse(open(os.path.join(folder, file)))
            text, tags = '', []
            for child in root.getroot():
                if child.tag == 'TEXT':
                    text = child.text
                if child.tag == 'TAGS':
                    for node in child:
                        tag_type = node.tag
                        tag_start = int(node.attrib['start'])
                        tag_end = int(node.attrib['end'])
                        tag_text = node.attrib['text']
                        compare_text = text[tag_start : tag_end]
                        tags.append([tag_type, tag_start, tag_end, tag_text])
                        if ' '.join(compare_text.split()) != ' '.join(tag_text.split()):
                            if '&' in compare_text:
                                continue
                            print('ERROR: Annotation offsets seem to be incorrect!')
            for node in reversed(tags):
                tokens = word_tokenize(node[-1])
                tokens = [x + '||' + 'I-' + node[0] for x in tokens]
                tokens[0] = tokens[0].replace('||I-', '||B-')
                text = text[:node[1]] + ' ' + ' '.join(tokens) + ' ' + text[node[2]:]
            texts.append(text)
        return texts

    def construct_examples(self, texts):
        examples = []
        ex_id = 0
        for text in texts:
            sents = sent_tokenize(text)
            for sent in sents:
                words = word_tokenize(sent)
                tokens = []
                labels = []
                for word in words:
                    if '||' not in word:
                        tokens.append(word)
                        labels.append('OUT')
                    else:
                        token, label = word.split('||')
                        tokens.append(token)
                        labels.append(label)
                examples.append({'id': ex_id, 'tokens': tokens, 'labels': labels})
                ex_id += 1
        return examples

    def correct_labels(self, examples):
        label_dict = {
            'DATE': 'MISC', 
            'PROFESSION': 'MISC', 
            'CONTACT': 'MISC', 
            'LOCATION': 'LOC', 
            'NAME': 'PER', 
            'PHI': 'MISC', 
            'ID': 'MISC', 
            'AGE': 'MISC'
        }
        for example in examples:
            corrected_labels = []
            for label in example['labels']:
                if '-' not in label:
                    corrected_labels.append(label)
                else:
                    start, tag = label.split('-')
                    corrected_labels.append('{}-{}'.format(start, label_dict[tag]))
            example['labels'] = corrected_labels
        return examples


# Dataloader for Timebank and MTSamples event extraction datasets
class EventTSVReader:

    def __init__(self, folder, task_string):
        train_ids = open(os.path.join(folder, 'train_ids.txt')).read().splitlines()
        dev_ids = open(os.path.join(folder, 'dev_ids.txt')).read().splitlines()
        test_ids = open(os.path.join(folder, 'test_ids.txt')).read().splitlines()

        examples = self.read_examples_from_files(folder)
        train_examples, dev_examples, test_examples = [], [], []
        for file_id in examples:
            if file_id in train_ids:
                train_examples += examples[file_id]
            if file_id in dev_ids:
                dev_examples += examples[file_id]
            if file_id in test_ids:
                test_examples += examples[file_id]
        
        # Dataset construction
        self.dataset = SeqDataset(train_examples, dev_examples, test_examples, task_string)

    def read_examples_from_files(self, folder):
        examples = {}
        ex_id = 0
        for file in os.listdir(folder):
            examples[file.split('.tsv')[0]] = []
            tokens, labels = [], []
            reader = open(os.path.join(folder, file))
            for line in reader:
                if line == '\n':
                    if tokens:
                        examples[file.split('.tsv')[0]].append({'id': ex_id, 'tokens': tokens, 'labels': labels})
                        ex_id += 1
                        tokens, labels = [], []
                else:
                    if '\t' not in line:
                        continue
                    if len(line.strip().split('\t')) == 1:  # Some misaligned entity annotations in MTSamples cause this
                        continue
                    token, label = line.strip().split('\t')
                    tokens.append(token)
                    labels.append(label if label != 'ENT' else 'O')
            # Last example may be left out
            if tokens:
                examples[file.split('.tsv')[0]].append({'id': ex_id, 'tokens': tokens, 'labels': labels})
        return examples


# Dataloader for i2b2 2012 event extraction dataset
class i2b22012Reader:

    def __init__(self, train_folder, test_folder, task_string):
        train_examples = self.read_examples_from_files(train_folder)
        test_examples = self.read_examples_from_files(test_folder)
        random.shuffle(train_examples)

        dev_size = int(math.ceil(0.1 * len(train_examples)))
        dev_examples = train_examples[:dev_size]
        train_examples = train_examples[dev_size:]

        # Dataset construction
        self.dataset = SeqDataset(train_examples, dev_examples, test_examples, task_string)

    def read_examples_from_files(self, folder):
        examples = []
        ex_id = 0
        for file in os.listdir(folder):
            if not file.endswith('.extent'):
                continue
            file_id = file.split('.xml')[0]
            text_reader = open(os.path.join(folder, file_id + '.xml.txt'))
            lines = []
            for line in text_reader:
                lines.append(line.strip().split())
            anno_reader = open(os.path.join(folder, file_id + '.xml.extent'), 'r')
            events = [['O']*len(i) for i in lines]
            for line in anno_reader:
                if not line.startswith('EVENT'):
                    continue
                event, _, _, _ = line.split('||')
                _, event_text, event_offsets = event.split('"')
                start_offset, end_offset = event_offsets.strip().split()
                start_sent, start_token = start_offset.split(':')
                end_sent, end_token = end_offset.split(':')
                start_sent, start_token, end_sent, end_token = int(start_sent), int(start_token), int(end_sent), int(end_token)
                if ' '.join(lines[start_sent-1][start_token : end_token+1]) != event_text:
                    print('ERROR: Annotation offsets seem to be incorrect!')
                events[start_sent-1][start_token : end_token+1] = ['EVENT'] * (end_token+1 - start_token)
            for token_list, event_list in zip(lines, events):
                examples.append({'id': ex_id, 'tokens': token_list, 'labels': event_list})
                ex_id += 1
        return examples


# Dataloader for file containing unlabeled data
class UnlabeledDataset:

    def __init__(self, text_file):

        reader = open(text_file)
        self.text = []
        for line in reader:
            self.text.append(line.strip())
        print('Loaded {} lines'.format(len(self.text)))

    def batch_and_tokenize_data(self, examples, tokenizer, batch_size):
        random.shuffle(examples)
        batches = []
        for i in range(0, len(examples), batch_size):
            start = i
            end = min(start+batch_size, len(examples))
            batch = self.tokenize_examples(examples[start:end], tokenizer)
            batches.append(batch)
        return batches

    def tokenize_examples(self, examples, tokenizer):
        tokenized_inputs = tokenizer(
            examples,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        return tokenized_inputs


# Class to mix labeled and unlabeled data for domain classification in Instance Weighting and Adversarial Training
class CombinedDataset:

    def __init__(self, source_sents, target_sents):
        min_len = min(len(source_sents), len(target_sents))
        random.shuffle(source_sents)
        random.shuffle(target_sents)
        source_data = source_sents[:min_len]
        target_data = target_sents[:min_len]
        self.domain_labels = [0] * len(source_data) + [1] * len(target_data)
        self.domain_data = source_data + target_data

    def batch_and_tokenize_data(self, tokenizer, batch_size):
        combined_data = list(zip(self.domain_data, self.domain_labels))
        random.shuffle(combined_data)
        domain_data, domain_labels = zip(*combined_data)
        domain_data, domain_labels = list(domain_data), list(domain_labels)
        batches = []
        for i in range(0, len(domain_data), batch_size):
            start = i
            end = min(start+batch_size, len(domain_data))
            batch = self.tokenize_examples(domain_data[start:end], domain_labels[start:end], tokenizer)
            batches.append(batch)
        return batches

    def tokenize_examples(self, examples, labels, tokenizer):
        tokenized_inputs = tokenizer(
            examples,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        tokenized_inputs["labels"] = torch.LongTensor(labels)
        return tokenized_inputs

# Class to store sequence labeling dataset (either for NER or event extraction)
class SeqDataset:

    def __init__(self, train_data, dev_data, test_data, task):

        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.task = task

        if task == 'ner-fine':
            self.label_vocab = {
                'B-PER': 0,
                'I-PER': 1,
                'B-LOC': 2,
                'I-LOC': 3,
                'B-ORG': 4,
                'I-ORG': 5,
                'B-MISC': 6,
                'I-MISC': 7,
                'OUT': 8
            }
        elif task == 'ee':
            self.label_vocab = {'EVENT': 1, 'O': 0}
        elif task == 'ner-coarse':
            self.label_vocab = {
                'B-ENT': 0,
                'I-ENT': 1,
                'OUT': 2
            }
        else:
            print('Task must be either NER (fine or coarse) or event extraction!!')
            exit(1)

        self.construct_label_sequences(self.train_data)
        self.construct_label_sequences(self.dev_data)
        self.construct_label_sequences(self.test_data)

    def construct_label_sequences(self, examples):
        if self.task != 'ner-coarse':
            for example in examples:
                label_seq = [self.label_vocab[x] for x in example['labels']]
                example['gold_seq'] = label_seq
        else:
            for example in examples:
                label_seq = []
                labels = []
                for label in example['labels']:
                    if label.startswith('O'):
                        label_seq.append(self.label_vocab['OUT'])
                        labels.append('OUT')
                    elif label.startswith('B'):
                        label_seq.append(self.label_vocab['B-ENT'])
                        labels.append('B-ENT')
                    elif label.startswith('I'):
                        label_seq.append(self.label_vocab['I-ENT'])
                        labels.append('I-ENT')
                example['gold_seq'] = label_seq
                example['labels'] = labels

    def batch_and_tokenize_data(self, tokenizer, batch_size):
        final_batches = []
        self.tokenized_examples = []
        for examples in [self.train_data, self.dev_data, self.test_data]:
            random.shuffle(examples)
            self.tokenized_examples.append(examples)
            batches = []
            for i in range(0, len(examples), batch_size):
                start = i
                end = min(start+batch_size, len(examples))
                batch = self.tokenize_and_align_labels(examples[start:end], tokenizer)
                batches.append(batch)
            final_batches.append(batches)
        return final_batches

    def tokenize_and_align_labels(self, examples, tokenizer):
        example_texts = [x['tokens'] for x in examples]
        tokenized_inputs = tokenizer(
            example_texts,
            padding='max_length',
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            max_length=128,
            return_tensors='pt'
        )
        labels = []
        for i, example in enumerate(examples):
            label_seq = example['gold_seq']
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_seq[word_idx])
                # For the other tokens in a word, we set the label to -100, but we might want to change that?
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = torch.LongTensor(labels)
        tokenized_inputs["example_ids"] = [x['id'] for x in examples]
        return tokenized_inputs
