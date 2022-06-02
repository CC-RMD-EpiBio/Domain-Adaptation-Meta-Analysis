import random
import pickle
import os
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from baseline_trainers import extract_entities, compute_token_f1, compute_exact_f1, ZeroShotTrainer

class AdversarialTrainer:

    def __init__(self, lr, epochs, device, out_dir):
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.out_dir = out_dir

    def train(self, model, train_data, dev_data, domain_data, out_dir, label_vocab, source_vocab):
        label_list = [x[0] for x in list(sorted(label_vocab.items(), key=lambda x: x[1]))]
        adv_criterion = nn.CrossEntropyLoss()
        adv_optimizer = optim.Adam(model.adv_classifier.parameters(), lr=self.lr)
        seq_optimizer = optim.Adam(model.rep_model.parameters(), lr=self.lr)
        adv_scheduler = optim.lr_scheduler.ReduceLROnPlateau(adv_optimizer, patience=2)
        seq_scheduler = optim.lr_scheduler.ReduceLROnPlateau(seq_optimizer, patience=2)
        print(label_list)
        step = 0
        prev_dev_loss = 10000
        for epoch in range(self.epochs):
            model.train()
            random.shuffle(train_data)
            random.shuffle(domain_data)
            epoch_loss = 0.0
            adv_loss = 0.0
            for i, batch in enumerate(train_data):
                seq_optimizer.zero_grad()
                adv_optimizer.zero_grad()
                domain_outputs, ner_outputs, sequence_domain_outputs, ner_loss = model(batch, domain_data[i], self.device, 1)
                domain_labels = domain_data[i]['labels'].to(self.device)
                cur_adv_loss = adv_criterion(domain_outputs, domain_labels)
                adv_loss += cur_adv_loss.item()
                cur_adv_loss.backward()
                adv_optimizer.step()
                seq_optimizer.zero_grad()
                adv_optimizer.zero_grad()
                domain_outputs, ner_outputs, sequence_domain_outputs, ner_loss = model(batch, domain_data[i], self.device, 2)
                cur_seq_loss = ner_loss + adv_criterion(sequence_domain_outputs, torch.LongTensor([0] * batch['labels'].size()[0]).to(self.device))
                epoch_loss += ner_loss.item()
                cur_seq_loss.backward()
                seq_optimizer.step()
                step += 1
                if step%1000 == 0:
                    print('Completed {} training steps'.format(step))
            epoch_loss /= len(train_data)
            print('Training loss after epoch {}: {}'.format(epoch, epoch_loss))
            # Checkpoint model after each epoch anyway, in addition to storing best loss
            #torch.save({
            #    'epoch': epoch,
            #    'model_state_dict': model.state_dict(),
            #    'adv_optimizer_state_dict': adv_optimizer.state_dict(),
            #    'seq_optimizer_state_dict': seq_optimizer.state_dict(),
            #    'loss': epoch_loss,
            #}, os.path.join(out_dir, 'checkpoints/checkpoint_{}.pt'.format(epoch)))
            dev_loss = self.test(model, dev_data, label_vocab, source_vocab, epoch=epoch, return_loss=True)
            if dev_loss < prev_dev_loss:
                prev_dev_loss = dev_loss
                torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
            seq_scheduler.step(dev_loss)
            adv_scheduler.step(dev_loss)

    def test(self, model, dev_data, label_vocab, source_vocab, epoch=0, return_loss=False, dump_test_preds=False, tokenized_examples=None):
        model.eval()
        label_list = [x[0] for x in list(sorted(label_vocab.items(), key=lambda x: x[1]))]
        dev_loss = 0.0
        model_predictions = []
        gold_labels = []
        all_example_ids = []
        for batch in dev_data:
            all_example_ids += batch['example_ids']
            domain_outputs, ner_outputs, sequence_domain_outputs, ner_loss = model(batch, batch, self.device, 2)
            dev_loss += ner_loss.item()
            cur_preds = np.argmax(ner_outputs.detach().cpu().numpy(), axis=2)
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

class InstanceWeightingTrainer:

    def __init__(self, lr, epochs, device, out_dir):
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.out_dir = out_dir

    def train(self, model, train_data, dev_data, domain_data, out_dir, label_vocab, source_vocab):
        label_list = [x[0] for x in list(sorted(label_vocab.items(), key=lambda x: x[1]))]
        domain_criterion = nn.CrossEntropyLoss()
        seq_criterion = nn.CrossEntropyLoss(reduction='none')
        domain_optimizer = optim.Adam(model.domain_classifier.parameters(), lr=self.lr)
        seq_optimizer = optim.Adam(model.rep_model.parameters(), lr=self.lr)
        domain_scheduler = optim.lr_scheduler.ReduceLROnPlateau(domain_optimizer, patience=2)
        seq_scheduler = optim.lr_scheduler.ReduceLROnPlateau(seq_optimizer, patience=2)
        print(label_list)
        step = 0
        prev_dev_loss = 10000
        for epoch in range(int(self.epochs/5)):
            model.train()
            random.shuffle(domain_data)
            domain_loss = 0.0
            for batch in domain_data:
                domain_optimizer.zero_grad()
                seq_optimizer.zero_grad()
                domain_outputs = model(batch, self.device, 'domain')
                domain_labels = batch['labels'].to(self.device)
                cur_domain_loss = domain_criterion(domain_outputs, domain_labels)
                domain_loss += cur_domain_loss.item()
                cur_domain_loss.backward()
                domain_optimizer.step()
            domain_loss /= len(domain_data)
            print('Domain loss after epoch {}: {}'.format(epoch, domain_loss))
        for epoch in range(self.epochs):
            model.train()
            domain_loss = 0.0
            epoch_loss = 0.0
            random.shuffle(train_data)
            for batch in train_data:
                domain_optimizer.zero_grad()
                seq_optimizer.zero_grad()
                seq_outputs, seq_domains = model(batch, self.device, 'seqlabel')
                seq_domains = nn.Softmax(dim=1)(seq_domains.detach())
                target_domain_weights = seq_domains[:,1]
                target_domain_weight_sum = torch.sum(target_domain_weights)
                target_domain_weights /= target_domain_weight_sum
                target_domain_weights *= target_domain_weights.size()[0]  # Compute weights from domain classifier target domain probabilities
                target_domain_weights = target_domain_weights.unsqueeze(-1).repeat(1, seq_outputs[1].size()[1])  
                # Repeat weights along sequence length dimension
                target_domain_weights = target_domain_weights.view(-1)
                cur_epoch_loss = 0.0
                if batch['attention_mask'] is not None:
                    active_loss = batch['attention_mask'].view(-1) == 1
                    active_logits = seq_outputs[1].view(-1, len(label_list))
                    active_labels = torch.where(
                        active_loss, batch['labels'].view(-1), torch.tensor(seq_criterion.ignore_index).type_as(batch['labels'])
                    )
                    cur_epoch_loss = seq_criterion(active_logits, active_labels.to(self.device))
                else:
                    cur_epoch_loss = seq_criterion(seq_outputs[1].view(-1, self.num_labels), batch['labels'].view(-1).to(self.device))
                cur_epoch_loss *= target_domain_weights
                cur_epoch_loss = torch.mean(cur_epoch_loss)
                epoch_loss += cur_epoch_loss.item()
                cur_epoch_loss.backward()
                seq_optimizer.step()
                step += 1
                if step%1000 == 0:
                    print('Completed {} training steps'.format(step))
            epoch_loss /= len(train_data)
            print('Training loss after epoch {}: {}'.format(epoch, epoch_loss))
            # Update domain classifier model
            random.shuffle(domain_data)
            for batch in domain_data:
                domain_optimizer.zero_grad()
                seq_optimizer.zero_grad()
                domain_outputs = model(batch, self.device, 'domain')
                domain_labels = batch['labels'].to(self.device)
                cur_domain_loss = domain_criterion(domain_outputs, domain_labels)
                domain_loss += cur_domain_loss.item()
                cur_domain_loss.backward()
                domain_optimizer.step()
            domain_loss /= len(domain_data)
            print('Domain loss after epoch {}: {}'.format(epoch, domain_loss))
            # Checkpoint model after each epoch anyway, in addition to storing best loss
            #torch.save({
            #    'epoch': epoch,
            #    'model_state_dict': model.state_dict(),
            #    'domain_optimizer_state_dict': domain_optimizer.state_dict(),
            #    'seq_optimizer_state_dict': seq_optimizer.state_dict(),
            #    'loss': epoch_loss,
            #}, os.path.join(out_dir, 'checkpoints/checkpoint_{}.pt'.format(epoch)))
            dev_loss = self.test(model, dev_data, label_vocab, source_vocab, epoch=epoch, return_loss=True)
            if dev_loss < prev_dev_loss:
                prev_dev_loss = dev_loss
                torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
            seq_scheduler.step(dev_loss)
            domain_scheduler.step(dev_loss)

    def test(self, model, dev_data, label_vocab, source_vocab, epoch=0, return_loss=False, dump_test_preds=False, tokenized_examples=None):
        model.eval()
        label_list = [x[0] for x in list(sorted(label_vocab.items(), key=lambda x: x[1]))]
        dev_loss = 0.0
        model_predictions = []
        gold_labels = []
        all_example_ids = []
        for batch in dev_data:
            all_example_ids += batch['example_ids']
            seq_outputs, seq_domains = model(batch, self.device, 'seqlabel')
            dev_loss += seq_outputs[0].item()
            cur_preds = np.argmax(seq_outputs[1].detach().cpu().numpy(), axis=2)
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

class SelfTrainer:

    def __init__(self, lr, epochs, device, st_epochs, out_dir):
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.st_epochs = st_epochs
        self.out_dir = out_dir
        self.model_trainer = ZeroShotTrainer(lr, epochs, device, out_dir)

    def get_top_predictions(self, model, data, topk, batch_size):
        model.eval()
        all_input_ids = []
        all_attention_masks = []
        all_token_type_ids = []
        all_preds = []
        all_conf = []
        for batch in data:
            batch = {x:y.to(self.device) for x,y in batch.items() if x not in ['example_ids']}
            outputs = model(**batch).logits
            outputs = nn.Softmax(dim=2)(outputs)
            cur_preds = np.argmax(outputs.detach().cpu().numpy(), axis=2)
            cur_conf = np.max(outputs.detach().cpu().numpy(), axis=2)
            cur_example_conf = np.mean(cur_conf, axis=1)
            all_conf += cur_example_conf.tolist()
            all_preds += cur_preds.tolist()
            all_input_ids += batch['input_ids'].tolist()
            all_attention_masks += batch['attention_mask'].tolist()
            all_token_type_ids += batch['token_type_ids'].tolist()
        sorted_example_indices = list(reversed(sorted(range(len(all_conf)), key=lambda x: all_conf[x])))
        top_preds = [all_preds[i] for i in sorted_example_indices][:topk]
        top_input_ids = [all_input_ids[i] for i in sorted_example_indices][:topk]
        top_attention_masks = [all_attention_masks[i] for i in sorted_example_indices][:topk]
        top_token_type_ids = [all_token_type_ids[i] for i in sorted_example_indices][:topk]
        annotated_batches = self.construct_tensor_batches(batch_size, top_input_ids, top_attention_masks, top_token_type_ids, top_preds)
        rem_input_ids = [all_input_ids[i] for i in sorted_example_indices][topk:]
        rem_attention_masks = [all_attention_masks[i] for i in sorted_example_indices][topk:]
        rem_token_type_ids = [all_token_type_ids[i] for i in sorted_example_indices][topk:]
        unannotated_batches = self.construct_tensor_batches(batch_size, rem_input_ids, rem_attention_masks, rem_token_type_ids)
        return annotated_batches, unannotated_batches

    def construct_tensor_batches(self, batch_size, input_ids, attention_masks, token_type_ids, labels=None):
        batches = []
        for start in range(0, len(input_ids), batch_size):
            end = min(len(input_ids), start+batch_size)
            batch = {
                'input_ids': torch.LongTensor(input_ids[start:end]),
                'attention_mask': torch.FloatTensor(attention_masks[start:end]),
                'token_type_ids': torch.LongTensor(token_type_ids[start:end]), 
            }
            if labels is not None:
                batch['labels'] = torch.LongTensor(labels[start:end])
            batches.append(batch)
        return batches

    def train(self, model, train_data, dev_data, unlabeled_data, out_dir, label_vocab, source_vocab, batch_size):
        # Train model on source labeled data
        if not os.path.exists(os.path.join(out_dir, 'st_pass0')):
            os.makedirs(os.path.join(out_dir, 'st_pass0'))
        if not os.path.exists(os.path.join(out_dir, 'st_pass0', 'checkpoints')):
            os.makedirs(os.path.join(out_dir, 'st_pass0', 'checkpoints'))
        self.model_trainer.train(model, train_data, dev_data, os.path.join(out_dir, 'st_pass0'), label_vocab)
        num_unlabeled_examples = len(unlabeled_data) * batch_size
        topk = int(num_unlabeled_examples / self.st_epochs)
        for epoch in range(self.st_epochs):
            model.load_state_dict(torch.load(os.path.join(os.path.join(out_dir, 'st_pass{}'.format(epoch)), 'best_model.pt')))
            annotated_batches, unannotated_batches = self.get_top_predictions(model, unlabeled_data, topk, batch_size)
            new_train_data = train_data + annotated_batches
            random.shuffle(new_train_data)
            if not os.path.exists(os.path.join(out_dir, 'st_pass{}'.format(epoch+1))):
                os.makedirs(os.path.join(out_dir, 'st_pass{}'.format(epoch+1)))
            if not os.path.exists(os.path.join(out_dir, 'st_pass{}'.format(epoch+1), 'checkpoints')):
                os.makedirs(os.path.join(out_dir, 'st_pass{}'.format(epoch+1), 'checkpoints'))
            self.model_trainer.train(model, new_train_data, dev_data, os.path.join(out_dir, 'st_pass{}'.format(epoch+1)), label_vocab, source_vocab)

    def test(self, model, test_data, label_vocab, source_vocab, dump_test_preds=False, tokenized_examples=None):
        return self.model_trainer.test(model, test_data, label_vocab, source_vocab, dump_test_preds=dump_test_preds, tokenized_examples=tokenized_examples)

class FeasyTrainer:

    def __init__(self, lr, epochs, device, out_dir):
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.out_dir = out_dir

    def train(self, model, train_data, dev_data, out_dir, label_vocab, train_domains, dev_domains, source_vocab):
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
            combined_train_data = list(zip(train_data, train_domains))
            random.shuffle(combined_train_data)
            train_data, train_domains = zip(*combined_train_data)
            epoch_loss = 0.0
            for index, batch in enumerate(train_data):
                optimizer.zero_grad()
                outputs = model(batch, self.device, train_domains[index])   # Batch is a dictionary from HF that needs to be unpacked
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
            dev_loss = self.test(model, dev_data, label_vocab, dev_domains, source_vocab, epoch=epoch, return_loss=True)
            if dev_loss < prev_dev_loss:
                prev_dev_loss = dev_loss
                torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
            scheduler.step(dev_loss)

    def test(self, model, dev_data, label_vocab, dev_domains, source_vocab, epoch=0, return_loss=False, dump_test_preds=False, tokenized_examples=None):
        model.eval()
        label_list = [x[0] for x in list(sorted(label_vocab.items(), key=lambda x: x[1]))]
        dev_loss = 0.0
        model_predictions = []
        gold_labels = []
        all_example_ids = []
        for index, batch in enumerate(dev_data):
            all_example_ids += batch['example_ids']
            outputs = model(batch, self.device, dev_domains[index])
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

class MultiTaskTrainer:

    def __init__(self, lr, epochs, device, out_dir):
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.out_dir = out_dir

    def train(self, model, train_data, dev_data, out_dir, label_vocab, train_domains, dev_domains, source_vocab):
        label_list = [x[0] for x in list(sorted(label_vocab.items(), key=lambda x: x[1]))]
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        print(label_list)
        task_sizes = Counter(train_domains)
        average_task_size = sum(list(task_sizes.values())) / len(set(list(task_sizes.keys())))
        task_weights = {x:float(average_task_size)/task_sizes[x] for x in task_sizes}
        # print('{} training batches found'.format(len(train_data)))
        # print('{} dev batches found'.format(len(dev_data)))
        step = 0
        prev_dev_loss = 10000
        for epoch in range(self.epochs):
            model.train()
            combined_train_data = list(zip(train_data, train_domains))
            random.shuffle(combined_train_data)
            train_data, train_domains = zip(*combined_train_data)
            epoch_loss = 0.0
            for index, batch in enumerate(train_data):
                optimizer.zero_grad()
                outputs = model(batch, self.device, train_domains[index])   # Batch is a dictionary from HF that needs to be unpacked
                loss = outputs[0]
                loss = loss * task_weights[train_domains[index]]
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                step += 1
                if step%1000 == 0:
                    print('Completed {} training steps'.format(step))
            epoch_loss /= len(train_data)
            print('Training loss after epoch {}: {}'.format(epoch, epoch_loss))
            # Checkpoint model after each epoch anyway, in addition to storing best loss
            # torch.save({
            #    'epoch': epoch,
            #    'model_state_dict': model.state_dict(),
            #    'optimizer_state_dict': optimizer.state_dict(),
            #    'loss': epoch_loss,
            #}, os.path.join(out_dir, 'checkpoints/checkpoint_{}.pt'.format(epoch)))
            dev_loss = self.test(model, dev_data, label_vocab, dev_domains, source_vocab, epoch=epoch, return_loss=True)
            if dev_loss < prev_dev_loss:
                prev_dev_loss = dev_loss
                torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
            scheduler.step(dev_loss)

    def test(self, model, dev_data, label_vocab, dev_domains, source_vocab, epoch=0, return_loss=False, dump_test_preds=False, tokenized_examples=None):
        model.eval()
        label_list = [x[0] for x in list(sorted(label_vocab.items(), key=lambda x: x[1]))]
        dev_loss = 0.0
        model_predictions = []
        gold_labels = []
        all_example_ids = []
        for index, batch in enumerate(dev_data):
            all_example_ids += batch['example_ids']
            outputs = model(batch, self.device, dev_domains[index])
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

class SupervisedInstanceWeightingTrainer:

    def __init__(self, lr, epochs, device, out_dir):
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.out_dir = out_dir
        self.model_trainer = ZeroShotTrainer(lr, epochs, device, out_dir)

    def construct_tensor_batches(self, batch_size, input_ids, attention_masks, token_type_ids, labels=None):
        batches = []
        for start in range(0, len(input_ids), batch_size):
            end = min(len(input_ids), start+batch_size)
            batch = {
                'input_ids': torch.LongTensor(input_ids[start:end]),
                'attention_mask': torch.FloatTensor(attention_masks[start:end]),
                'token_type_ids': torch.LongTensor(token_type_ids[start:end]),
            }
            if labels is not None:
                batch['labels'] = torch.LongTensor(labels[start:end])
            batches.append(batch)
        return batches

    def get_top_predictions(self, model, data, topk, batch_size):
        model.eval()
        all_input_ids = []
        all_attention_masks = []
        all_token_type_ids = []
        all_preds = []
        all_conf = []
        for batch in data:
            batch = {x:y.to(self.device) for x,y in batch.items() if x not in ['example_ids']}
            outputs = model(**batch).logits
            outputs = nn.Softmax(dim=2)(outputs)
            cur_preds = torch.argmax(outputs.detach(), dim=2)
            correct = torch.eq(cur_preds, batch['labels'])
            incorrect = 1 - correct.long()
            cur_conf = torch.max(outputs.detach(), dim=2)[0]
            cur_incorrect_conf = cur_conf * incorrect
            cur_example_conf = torch.mean(cur_incorrect_conf, dim=1)
            all_conf += cur_example_conf.tolist()
            all_preds += cur_preds.tolist()
            all_input_ids += batch['input_ids'].tolist()
            all_attention_masks += batch['attention_mask'].tolist()
            all_token_type_ids += batch['token_type_ids'].tolist()
        sorted_example_indices = list(sorted(range(len(all_conf)), key=lambda x: all_conf[x]))
        top_preds = [all_preds[i] for i in sorted_example_indices][:topk]
        top_input_ids = [all_input_ids[i] for i in sorted_example_indices][:topk]
        top_attention_masks = [all_attention_masks[i] for i in sorted_example_indices][:topk]
        top_token_type_ids = [all_token_type_ids[i] for i in sorted_example_indices][:topk]
        annotated_batches = self.construct_tensor_batches(batch_size, top_input_ids, top_attention_masks, top_token_type_ids, top_preds)
        return annotated_batches

    def train(self, model, source_train_data, source_dev_data, target_train_data, target_dev_data, out_dir, label_vocab, source_vocab, batch_size):
        label_list = [x[0] for x in list(sorted(label_vocab.items(), key=lambda x: x[1]))]
        print(label_list)
        self.model_trainer.train(model, target_train_data, target_dev_data, out_dir, label_vocab, source_vocab)
        model.load_state_dict(torch.load(os.path.join(out_dir, 'best_model.pt')))
        chosen_source_train_batches = self.get_top_predictions(model, source_train_data, len(target_train_data), batch_size)
        joint_train_batches = target_train_data + chosen_source_train_batches
        self.model_trainer.train(model, joint_train_batches, target_dev_data, out_dir, label_vocab, source_vocab)

    def test(self, model, test_data, label_vocab, source_vocab, epoch=0, return_loss=False, dump_test_preds=False, tokenized_examples=None):
        return self.model_trainer.test(model, test_data, label_vocab, source_vocab, dump_test_preds=dump_test_preds, tokenized_examples=tokenized_examples)
