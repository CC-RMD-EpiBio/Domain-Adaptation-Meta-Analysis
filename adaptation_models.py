import torch
import torch.nn as nn
from torch.autograd import Function, Variable

# Gradient-reversal layer
class GradReverse(Function):
    
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -1 * ctx.constant * grad_output, None

gradreverse = GradReverse.apply

# Domain detection classifier
class DomainClassifier(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(DomainClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = (2*n_layers) + 1
        
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])
        self.layers.append(nn.Tanh())
        for i in range(1, n_layers):
            self.layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, features):
        for i in range(self.n_layers):
            features = self.layers[i](features)
        return features

# Complete adversarial model
class AdversarialSequenceLabeler(nn.Module):

    def __init__(self, model, num_classes, adv_hidden_size, adv_layers, num_domains, adv_coeff, dropout):
        super(AdversarialSequenceLabeler, self).__init__()
        self.rep_model = model
        self.adv_classifier = DomainClassifier(self.rep_model.config.hidden_size, adv_hidden_size, num_domains, adv_layers)
        self.adv_coeff = adv_coeff

    def forward(self, ner_data, domain_data, device, pass_num):
        # Forward pass for loss 1: Only adversarial classifier weights
        if pass_num == 1:
            domain_data_nolabels = {x:y.to(device) for x,y in domain_data.items() if x not in ['labels', 'example_ids']}
            outputs = self.rep_model.bert(**domain_data_nolabels, output_hidden_states=True)
            domain_reps = torch.mean(outputs.hidden_states[-1], dim=1)
            domain_outputs = self.adv_classifier(domain_reps)
            return domain_outputs, None, None, None
        if pass_num == 2:
            ner_data = {x:y.to(device) for x,y in ner_data.items() if x not in ['example_ids']}   # Tensors need to be moved to GPU
            outputs = self.rep_model(**ner_data, output_hidden_states=True)
            ner_loss = outputs[0]
            ner_outputs = outputs[1]
            sequence_domain_reps = torch.mean(outputs[2][-1], dim=1)
            sequence_domain_outputs = self.adv_classifier(gradreverse(sequence_domain_reps, self.adv_coeff))
            return None, ner_outputs, sequence_domain_outputs, ner_loss

# Instance weighting model (for unlabeled setting)
class InstanceWeightingUnlabeled(nn.Module):

    def __init__(self, model, num_classes, dom_hidden_size, dom_layers, num_domains):
        super(InstanceWeightingUnlabeled, self).__init__()
        self.rep_model = model
        self.domain_classifier = DomainClassifier(self.rep_model.config.hidden_size, dom_hidden_size, num_domains, dom_layers)

    def forward(self, batch, device, run):
        if run == 'domain':
            domain_data_nolabels = {x:y.to(device) for x,y in batch.items() if x not in ['labels', 'example_ids']}
            outputs = self.rep_model.bert(**domain_data_nolabels, output_hidden_states=True)
            domain_reps = torch.mean(outputs.hidden_states[-1], dim=1)
            domain_outputs = self.domain_classifier(domain_reps)
            return domain_outputs
        if run == 'seqlabel':
            ner_data = {x:y.to(device) for x,y in batch.items() if x not in ['example_ids']}
            outputs = self.rep_model(**ner_data, output_hidden_states=True)
            sequence_domain_reps = torch.mean(outputs[2][-1], dim=1)
            sequence_domain_outputs = self.domain_classifier(sequence_domain_reps)
            return outputs, sequence_domain_outputs

# Feature augmentation model (for labeled setting)
class FeatureAugmentation(nn.Module):

    def __init__(self, model, num_labels, num_domains):
        super(FeatureAugmentation, self).__init__()
        self.rep_model = model
        self.num_labels = num_labels
        self.num_domains = num_domains
        self.classifier = nn.Linear((num_domains + 1) * model.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, batch, device, domain):
        gpu_batch = {x:y.to(device) for x,y in batch.items() if x not in ['labels', 'example_ids']}
        sequence_output = self.rep_model.bert(**gpu_batch)[0]
        zero_tensor = torch.zeros(sequence_output.size()).to(device)
        if domain == 0:
            augmented_output = torch.cat([sequence_output, zero_tensor, sequence_output], dim=-1)
        elif domain == 1:
            augmented_output = torch.cat([zero_tensor, sequence_output, sequence_output], dim=-1)
        logits = self.classifier(self.dropout(augmented_output))
        loss = None
        attention_mask = gpu_batch['attention_mask']
        labels = batch['labels'].to(device)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return [loss, logits]

# Multi-tasking model (loss augmentation for labeled setting)
class MultiTaskModel(nn.Module):

    def __init__(self, model, num_labels, num_domains):
        super(MultiTaskModel, self).__init__()
        self.rep_model = model
        self.num_labels = num_labels
        self.num_domains = num_domains
        self.classifiers = nn.ModuleList()
        for i in range(num_domains):
            self.classifiers.append(nn.Linear(model.config.hidden_size, num_labels))
        self.dropout = nn.Dropout(0.1)

    def forward(self, batch, device, domain):
        gpu_batch = {x:y.to(device) for x,y in batch.items() if x not in ['labels', 'example_ids']}
        sequence_output = self.rep_model.bert(**gpu_batch)[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifiers[domain](sequence_output)
        loss = None
        attention_mask = gpu_batch['attention_mask']
        labels = batch['labels'].to(device)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return [loss, logits]
