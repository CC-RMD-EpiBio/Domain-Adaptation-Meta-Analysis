import os
import pickle
import random
from collections import Counter
import numpy as np
import torch
import torch.optim as optim

def compute_token_f1(predictions, references, label_list, source_indicators):
    results = {}
    pred_count = 0.0
    ref_count = 0.0
    correct_count = 0.0
    iv_pred, oov_pred = 0.0, 0.0
    iv_ref, oov_ref = 0.0, 0.0
    iv_correct, oov_correct = 0.0, 0.0
    i = 0
    for pred_list, ref_list in zip(predictions, references):
        j = 0
        for pred, ref in zip(pred_list, ref_list):
            if pred == 'EVENT':
                pred_count += 1
                if isinstance(source_indicators, list):
                    if source_indicators[i][j] == 0:
                        iv_pred += 1
                    else:
                        oov_pred += 1
                if pred == ref:
                    correct_count += 1
                    if isinstance(source_indicators, list):
                        if source_indicators[i][j] == 0:
                            iv_correct += 1
                        else:
                            oov_correct += 1
            if ref == 'EVENT':
                ref_count += 1
                if isinstance(source_indicators, list):
                    if source_indicators[i][j] == 0:
                        iv_ref += 1
                    else:
                        oov_ref += 1
            j += 1
        i += 1
    rec = correct_count / ref_count if ref_count != 0 else 0.0
    prec = correct_count / pred_count if pred_count != 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if prec != 0 and rec != 0 else 0.0
    iv_rec = iv_correct / iv_ref if iv_ref != 0 else 0.0
    iv_prec = iv_correct / iv_pred if iv_pred != 0 else 0.0
    iv_f1 = (2 * iv_prec * iv_rec) / (iv_prec + iv_rec) if iv_prec != 0 and iv_rec != 0 else 0.0
    oov_rec = oov_correct / oov_ref if oov_ref != 0 else 0.0
    oov_prec = oov_correct / oov_pred if oov_pred != 0 else 0.0
    oov_f1 = (2 * oov_prec * oov_rec) / (oov_prec + oov_rec) if oov_prec != 0 and oov_rec != 0 else 0.0
    results['overall_f1'] = f1
    results['overall_prec'] = prec
    results['overall_rec'] = rec
    print('In-Vocabulary F1: {}'.format(iv_f1))
    print('Out-of-Vocabulary F1: {}'.format(oov_f1))
    return results

def compute_exact_f1(predictions, references, label_list, source_indicators):
    results = {}
    pred_count = 0.0
    ref_count = 0.0
    correct_count = 0.0
    iv_pred, oov_pred = 0.0, 0.0
    iv_ref, oov_ref = 0.0, 0.0
    iv_correct, oov_correct = 0.0, 0.0
    num_tags = (len(label_list)-1)/2
    per_tag_pred = Counter()
    per_tag_ref = Counter()
    per_tag_correct = Counter()
    i = 0
    for pred_list, ref_list in zip(predictions, references):
        pred_entities = extract_entities(pred_list)
        ref_entities = extract_entities(ref_list)
        pred_count += len(pred_entities)
        if isinstance(source_indicators, list):
            for entity in pred_entities:
                start, end = entity
                is_oov = sum(source_indicators[i][start:end+1])
                if is_oov > 0:
                    oov_pred += 1
                else:
                    iv_pred += 1
            for entity in ref_entities:
                start, end = entity
                is_oov = sum(source_indicators[i][start:end+1])
                if is_oov > 0:
                    oov_ref += 1
                else:
                    iv_ref += 1
        ref_count += len(ref_entities)
        per_tag_pred.update(list(pred_entities.values()))
        per_tag_ref.update(list(ref_entities.values()))
        matched_spans = set(pred_entities.keys()).intersection(set(ref_entities.keys()))# Find entities that match boundaries exactly
        for span in matched_spans:
            if pred_entities[span] == ref_entities[span]:  # Check that type also matches
                correct_count += 1
                if isinstance(source_indicators, list):
                    start, end = span
                    is_oov = sum(source_indicators[i][start:end+1])
                    if is_oov > 0:
                        oov_correct += 1
                    else:
                        iv_correct += 1
                per_tag_correct.update([pred_entities[span]])
        i += 1
    #print(per_tag_ref)
    #exit(1)
    rec = correct_count / ref_count if ref_count != 0 else 0.0
    prec = correct_count / pred_count if pred_count != 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if prec != 0 and rec != 0 else 0.0
    iv_rec = iv_correct / iv_ref if iv_ref != 0 else 0.0
    iv_prec = iv_correct / iv_pred if iv_pred != 0 else 0.0
    iv_f1 = (2 * iv_prec * iv_rec) / (iv_prec + iv_rec) if iv_prec != 0 and iv_rec != 0 else 0.0
    oov_rec = oov_correct / oov_ref if oov_ref != 0 else 0.0
    oov_prec = oov_correct / oov_pred if oov_pred != 0 else 0.0
    oov_f1 = (2 * oov_prec * oov_rec) / (oov_prec + oov_rec) if oov_prec != 0 and oov_rec != 0 else 0.0
    print('In-Vocabulary F1: {}'.format(iv_f1))
    print('Out-of-Vocabulary F1: {}'.format(oov_f1))
    for label in per_tag_ref:
        tag_prec = per_tag_correct[label] / float(per_tag_pred[label]) if per_tag_pred[label] != 0 else 0.0
        tag_rec = per_tag_correct[label] / float(per_tag_ref[label]) if per_tag_ref[label] != 0 else 0.0
        tag_f1 = (2 * tag_prec * tag_rec) / (tag_prec + tag_rec) if tag_rec != 0 and tag_prec != 0 else 0.0
        results[label] = tag_f1
    results['overall_f1'] = f1
    results['overall_prec'] = prec
    results['overall_rec'] = rec
    return results

def extract_entities(sequence):
    entities = {}
    starts = [i for i,x in enumerate(sequence) if x.startswith('B')]
    ends = []
    for idx in starts:
        if idx == len(sequence)-1 or not sequence[idx+1].startswith('I'):
            ends.append(idx)
            continue
        cur_idx = idx + 1
        while cur_idx < len(sequence) and sequence[cur_idx].startswith('I'):
            cur_idx += 1
        ends.append(cur_idx-1)
    if len(starts) != len(ends):
        print('Missing end indices for some predictions!!')
    offsets = list(zip(starts, ends))
    for offset in offsets:
        tag = sequence[offset[0]].split('-')[1]
        entities[offset] = tag
    return entities

class ZeroShotTrainer:

    def __init__(self, lr, epochs, device, out_dir):
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.out_dir = out_dir

    def train(self, model, train_data, dev_data, out_dir, label_vocab, source_vocab):
        label_list = [x[0] for x in list(sorted(label_vocab.items(), key=lambda x: x[1]))]
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        print(label_list)
        # print('{} training batches found'.format(len(train_data)))
        # print('{} dev batches found'.format(len(dev_data)))
        step = 0
        prev_dev_loss = 10000
        for epoch in range(self.epochs):
            model.train()
            random.shuffle(train_data)
            epoch_loss = 0.0
            for batch in train_data:
                optimizer.zero_grad()
                batch = {x:y.to(self.device) for x,y in batch.items() if x not in ['example_ids']}   # Tensors need to be moved to GPU
                outputs = model(**batch)   # Batch is a dictionary from HF that needs to be unpacked
                loss = outputs[0]
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                step += 1
                if step%1000 == 0:
                    print('Completed {} training steps'.format(step))
            epoch_loss /= len(train_data)
            print('Training loss after epoch {}: {}'.format(epoch, epoch_loss))
            # Checkpoint model after each epoch anyway, in addition to storing best loss
            #torch.save({
            #    'epoch': epoch,
            #    'model_state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'loss': epoch_loss,
            #}, os.path.join(out_dir, 'checkpoints/checkpoint_{}.pt'.format(epoch)))
            dev_loss = self.test(model, dev_data, label_vocab, source_vocab, epoch=epoch, return_loss=True)
            if dev_loss < prev_dev_loss:
                prev_dev_loss = dev_loss
                torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
            scheduler.step(dev_loss)

    def test(self, model, dev_data, label_vocab, source_vocab, epoch=0, return_loss=False, dump_test_preds=False, tokenized_examples=None):
        model.eval()
        label_list = [x[0] for x in list(sorted(label_vocab.items(), key=lambda x: x[1]))]
        dev_loss = 0.0
        model_predictions = []
        gold_labels = []
        all_example_ids = []
        for batch in dev_data:
            all_example_ids += batch['example_ids']
            batch = {x:y.to(self.device) for x,y in batch.items() if x not in ['example_ids']}
            outputs = model(**batch)
            loss = outputs[0]
            dev_loss += loss.item()
            cur_preds = np.argmax(outputs[1].detach().cpu().numpy(), axis=2)
            labels = batch['labels'].cpu().numpy()
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(cur_preds, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(cur_preds, labels)
            ]
            model_predictions += true_predictions
            gold_labels += true_labels
        if dump_test_preds:
            example_dump = {}
            for ex_id, predictions in zip(all_example_ids, model_predictions):
                example_dump[ex_id] = {'preds': predictions}
            for example in tokenized_examples:
                example_dump[example['id']]['tokens'] = example['tokens']
                example_dump[example['id']]['labels'] = example['labels']
            pickle.dump(example_dump, open(os.path.join(self.out_dir, 'target_test_predictions.pkl'), 'wb'))
        dev_loss /= len(dev_data)
        print('Validation loss after epoch {}: {}'.format(epoch, dev_loss))
        results = {}
        if len(label_list) > 2:
            results = compute_exact_f1(model_predictions, gold_labels, label_list, source_vocab)
            print('------------------Scores for Epoch {}-------------------'.format(epoch))
            print('Overall Exact Precision: {}'.format(results['overall_prec']))
            print('Overall Exact Recall: {}'.format(results['overall_rec']))
            print('Overall Exact F1 Score: {}'.format(results['overall_f1']))
            for label in results:
                if 'overall' in label:
                    continue
                print('F1 Score for {}: {}'.format(label, results[label]))
        else:
            results = compute_token_f1(model_predictions, gold_labels, label_list, source_vocab)
            print('------------------Scores for Epoch {}-------------------'.format(epoch))
            print('Overall Token Precision: {}'.format(results['overall_prec']))
            print('Overall Token Recall: {}'.format(results['overall_rec']))
            print('Overall Token F1 Score: {}'.format(results['overall_f1']))
        if return_loss:
            return dev_loss
        return results
