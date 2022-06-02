import os
import argparse
import random
import numpy as np
import torch
import pickle

from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, \
        PreTrainedTokenizerFast, set_seed
from dataloader import CoNLL2003Reader, i2b22006Reader, i2b22014Reader, \
        i2b22012Reader, EventTSVReader, UnlabeledDataset, CombinedDataset
from baseline_trainers import ZeroShotTrainer
from adaptation_models import AdversarialSequenceLabeler, InstanceWeightingUnlabeled, FeatureAugmentation, MultiTaskModel
from adaptation_trainers import AdversarialTrainer, InstanceWeightingTrainer, SelfTrainer, \
        FeasyTrainer, MultiTaskTrainer, SupervisedInstanceWeightingTrainer

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, action='store', required=True, \
            help='Source dataset [conll2003, i2b22006, i2b22010, i2b22012, i2b22014, timebank, mtsamples]')
    parser.add_argument('--target', type=str, action='store', required=True, \
            help='Target dataset [conll2003, i2b22006, i2b22010, i2b22012, i2b22014, timebank, mtsamples]')
    parser.add_argument('--task', type=str, action='store', required=True, help='Choose task from [ee, ner-fine, ner-coarse]')
    parser.add_argument('--out_dir', type=str, action='store', required=True, help='Path to store model output')
    parser.add_argument('--level', type=int, action='store', default=2, help='Level of data scarcity [0-3]')
    parser.add_argument('--model_name_or_path', type=str, action='store', required=True, help='Pretrained model to use')
    parser.add_argument('--device', type=str, action='store', required=True, help='ID of GPU device to use')
    parser.add_argument('--da_method', type=str, action='store', help='Adaptation method to use from [LA, FA, PL, PT, IW]')
    parser.add_argument('--lr', type=float, action='store', default=2e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, action='store', default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, action='store', default=8, help='Batch size')
    parser.add_argument('--seed', type=int, action='store', default=42, help='Random seed')
    parser.add_argument('--adv_hidden_size', type=int, action='store', default=300, help='Hidden size for adversarial classifier')
    parser.add_argument('--adv_layers', type=int, action='store', default=1, help='Number of layers for adversarial classifier')
    parser.add_argument('--adv_coeff', type=float, action='store', default=1.0, help='Balancing coefficient for adversarial training')
    parser.add_argument('--dropout', type=float, action='store', default=0.1, help='Dropout')
    parser.add_argument('--pt_model_name_or_path', type=str, action='store', help='Unsupervised pretrained model to use')
    parser.add_argument('--pl_passes', type=str, action='store', default=3, help='Number of self-training passes to run')
    args = parser.parse_args()
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    checkpoint_dir = os.path.join(args.out_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    source_dataset = read_dataset(args.source, args.task)
    target_dataset = read_dataset(args.target, args.task)
    source_vocab = set()
    for example in source_dataset.train_data + source_dataset.dev_data:
        source_vocab.update(example['tokens'])    
    
    all_results = {}
    for num_examples in range(100, 1001, 100):
        all_results[num_examples] = {}
        cur_out_dir = os.path.join(args.out_dir, 'datasize_{}'.format(num_examples))
        if not os.path.exists(cur_out_dir):
            os.makedirs(cur_out_dir)
        num_train_examples = 0.9 * num_examples
        num_dev_examples = 0.1 * num_examples
        target_dataset.train_data = target_dataset.train_data[:900]
        target_dataset.dev_data = target_dataset.dev_data[:100]

        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=len(source_dataset.label_vocab),
            label2id=source_dataset.label_vocab,
            id2label={i: l for l, i in source_dataset.label_vocab.items()},
            finetuning_task='ner',
            cache_dir='../cache/',
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir='../cache/',
            use_fast=True,
        )

        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir='../cache/',
        )
        device = torch.device(args.device)
        model = model.to(device)

        source_train_batches, source_dev_batches, source_test_batches = source_dataset.batch_and_tokenize_data(tokenizer, args.batch_size)
        target_train_batches, target_dev_batches, target_test_batches = target_dataset.batch_and_tokenize_data(tokenizer, args.batch_size)
        source_oov_indicators, target_oov_indicators = [], []
        for example in source_dataset.tokenized_examples[-1]:
            source_oov_indicators.append([0 if x in source_vocab else 1 for x in example['tokens']])
        for example in target_dataset.tokenized_examples[-1]:
            target_oov_indicators.append([0 if x in source_vocab else 1 for x in example['tokens']])
        # Subsample for testing
        # source_train_batches = source_train_batches[:10]
        # source_dev_batches = source_dev_batches[:10]
        # source_test_batches = source_test_batches[:10]
        '''
        if args.level == 0:
            trainer = ZeroShotTrainer(args.lr, args.epochs, device)
            trainer.train(model, source_train_batches, source_dev_batches, args.out_dir, source_dataset.label_vocab)
            model.load_state_dict(torch.load(os.path.join(args.out_dir, 'best_model.pt')))
            print('Scores on source testing data')
            trainer.test(model, source_test_batches, source_dataset.label_vocab)
            print('Scores on target testing data')
            trainer.test(model, target_test_batches, source_dataset.label_vocab)

        if args.level == 1:
            unlabeled_dataset = UnlabeledDataset('../datasets/mimiciii_notes_raw.txt')
            labeled_dataset = [' '.join(x['tokens']) for x in source_dataset.train_data]
            print('{} labeled lines'.format(len(labeled_dataset)))
            if args.da_method == 'LA':
                domain_dataset = CombinedDataset(labeled_dataset, unlabeled_dataset.text)
                domain_batches = domain_dataset.batch_and_tokenize_data(tokenizer, args.batch_size)
                adv_model = AdversarialSequenceLabeler(model, len(source_dataset.label_vocab), args.adv_hidden_size, \
                    args.adv_layers, 2, args.adv_coeff, args.dropout)
                adv_model = adv_model.to(device)
                trainer = AdversarialTrainer(args.lr, args.epochs, device)
                trainer.train(adv_model, source_train_batches, source_dev_batches, domain_batches, args.out_dir, source_dataset.label_vocab)
                adv_model.load_state_dict(torch.load(os.path.join(args.out_dir, 'best_model.pt')))
                print('Scores on source testing data')
                trainer.test(adv_model, source_test_batches, source_dataset.label_vocab)
                print('Scores on target testing data')
                trainer.test(adv_model, target_test_batches, source_dataset.label_vocab)
            if args.da_method == 'IW':
                domain_dataset = CombinedDataset(labeled_dataset, unlabeled_dataset.text)
                domain_batches = domain_dataset.batch_and_tokenize_data(tokenizer, args.batch_size)
                iw_model = InstanceWeightingUnlabeled(model, len(source_dataset.label_vocab), args.adv_hidden_size, args.adv_layers, 2)
                iw_model = iw_model.to(device)
                trainer = InstanceWeightingTrainer(args.lr, args.epochs, device)
                trainer.train(iw_model, source_train_batches, source_dev_batches, domain_batches, args.out_dir, source_dataset.label_vocab)
                iw_model.load_state_dict(torch.load(os.path.join(args.out_dir, 'best_model.pt')))
                print('Scores on source testing data')
                trainer.test(iw_model, source_test_batches, source_dataset.label_vocab)
                print('Scores on target testing data')
                trainer.test(iw_model, target_test_batches, source_dataset.label_vocab)
            if args.da_method == 'PL':
                random.shuffle(unlabeled_dataset.text)
                chosen_unlabeled_sents = unlabeled_dataset.text[:100000]
                unlabeled_batches = unlabeled_dataset.batch_and_tokenize_data(chosen_unlabeled_sents, tokenizer, args.batch_size)
                trainer = SelfTrainer(args.lr, args.epochs, device, args.pl_passes)
                trainer.train(model, source_train_batches, source_dev_batches, unlabeled_batches, args.out_dir, source_dataset.label_vocab, args.batch_size)
                model.load_state_dict(torch.load(os.path.join(args.out_dir, 'st_pass{}'.format(args.pl_passes), 'best_model.pt')))
                print('Scores on source testing data')
                trainer.test(model, source_test_batches, source_dataset.label_vocab)
                print('Scores on target testing data')
                trainer.test(model, target_test_batches, source_dataset.label_vocab)
            if args.da_method == 'PT':
                model = AutoModelForTokenClassification.from_pretrained(
                    args.pt_model_name_or_path,
                    config=config,
                    cache_dir='../cache/',
                )
                model = model.to(device)
                trainer = ZeroShotTrainer(args.lr, args.epochs, device)
                trainer.train(model, source_train_batches, source_dev_batches, args.out_dir, source_dataset.label_vocab)
                model.load_state_dict(torch.load(os.path.join(args.out_dir, 'best_model.pt')))
                print('Scores on source testing data')
                trainer.test(model, source_test_batches, source_dataset.label_vocab)
                print('Scores on target testing data')
                trainer.test(model, target_test_batches, source_dataset.label_vocab)

        if args.level == 2:
        '''
        if args.da_method == 'TGONLY':
            trainer = ZeroShotTrainer(args.lr, args.epochs, device, args.out_dir)
            trainer.train(model, target_train_batches, target_dev_batches, cur_out_dir, source_dataset.label_vocab, None)
            model.load_state_dict(torch.load(os.path.join(cur_out_dir, 'best_model.pt')))
            print('Scores on source testing data')
            all_results[num_examples]['source'] = trainer.test(model, source_test_batches, source_dataset.label_vocab, None)
            print('Scores on target testing data')
            all_results[num_examples]['target'] = trainer.test(model, target_test_batches, source_dataset.label_vocab, None)
        if args.da_method == 'SCTGCL':
            trainer = ZeroShotTrainer(args.lr, args.epochs, device, args.out_dir)
            trainer.train(model, source_train_batches, source_dev_batches, cur_out_dir, source_dataset.label_vocab, None)
            trainer.train(model, target_train_batches, target_dev_batches, cur_out_dir, source_dataset.label_vocab, None)
            model.load_state_dict(torch.load(os.path.join(cur_out_dir, 'best_model.pt')))
            print('Scores on source testing data')
            all_results[num_examples]['source'] = trainer.test(model, source_test_batches, source_dataset.label_vocab, None)
            print('Scores on target testing data')
            all_results[num_examples]['target'] = trainer.test(model, target_test_batches, source_dataset.label_vocab, None)
        if args.da_method == 'SCTGJL':
            trainer = ZeroShotTrainer(args.lr, args.epochs, device, args.out_dir)
            joint_train_batches = source_train_batches + target_train_batches
            joint_dev_batches = source_dev_batches + target_dev_batches
            trainer.train(model, joint_train_batches, joint_dev_batches, cur_out_dir, source_dataset.label_vocab, None)
            model.load_state_dict(torch.load(os.path.join(cur_out_dir, 'best_model.pt')))
            print('Scores on source testing data')
            all_results[num_examples]['source'] = trainer.test(model, source_test_batches, source_dataset.label_vocab, None)
            print('Scores on target testing data')
            all_results[num_examples]['target'] = trainer.test(model, target_test_batches, source_dataset.label_vocab, None)
        if args.da_method == 'FA':
            model = FeatureAugmentation(model, len(source_dataset.label_vocab), 2)
            model.to(device)
            joint_train_domains = [0] * len(source_train_batches) + [1] * len(target_train_batches)
            joint_dev_domains = [0] * len(source_dev_batches) + [1] * len(target_dev_batches)
            joint_train_batches = source_train_batches + target_train_batches
            joint_dev_batches = source_dev_batches + target_dev_batches
            trainer = FeasyTrainer(args.lr, args.epochs, device, args.out_dir)
            trainer.train(model, joint_train_batches, joint_dev_batches, cur_out_dir, source_dataset.label_vocab, joint_train_domains, joint_dev_domains, None)
            model.load_state_dict(torch.load(os.path.join(cur_out_dir, 'best_model.pt')))
            print('Scores on source testing data')
            all_results[num_examples]['source'] = trainer.test(model, source_test_batches, source_dataset.label_vocab, [0]*len(source_test_batches), None)
            print('Scores on target testing data')
            all_results[num_examples]['target'] = trainer.test(model, target_test_batches, source_dataset.label_vocab, [1]*len(target_test_batches), None)
        if args.da_method == 'LA':
            model = MultiTaskModel(model, len(source_dataset.label_vocab), 2)
            model.to(device)
            joint_train_domains = [0] * len(source_train_batches) + [1] * len(target_train_batches)
            joint_dev_domains = [0] * len(source_dev_batches) + [1] * len(target_dev_batches)
            joint_train_batches = source_train_batches + target_train_batches
            joint_dev_batches = source_dev_batches + target_dev_batches
            trainer = MultiTaskTrainer(args.lr, args.epochs, device, args.out_dir)
            trainer.train(model, joint_train_batches, joint_dev_batches, cur_out_dir, source_dataset.label_vocab, joint_train_domains, joint_dev_domains, None)
            model.load_state_dict(torch.load(os.path.join(cur_out_dir, 'best_model.pt')))
            print('Scores on source testing data')
            all_results[num_examples]['source'] = trainer.test(model, source_test_batches, source_dataset.label_vocab, [0]*len(source_test_batches), None)
            print('Scores on target testing data')
            all_results[num_examples]['target'] = trainer.test(model, target_test_batches, source_dataset.label_vocab, [1]*len(target_test_batches), None)
        if args.da_method == 'IW':
            trainer = SupervisedInstanceWeightingTrainer(args.lr, args.epochs, device, args.out_dir)
            trainer.train(model, source_train_batches, source_dev_batches, target_train_batches, target_dev_batches, \
                    cur_out_dir, source_dataset.label_vocab, None, args.batch_size)
            model.load_state_dict(torch.load(os.path.join(cur_out_dir, 'best_model.pt')))
            print('Scores on source testing data')
            all_results[num_examples]['source'] = trainer.test(model, source_test_batches, source_dataset.label_vocab, None)
            print('Scores on target testing data')
            all_results[num_examples]['target'] = trainer.test(model, target_test_batches, source_dataset.label_vocab, None)
    
    pickle.dump(all_results, open(os.path.join(args.out_dir, 'results.pkl'), 'wb'))
