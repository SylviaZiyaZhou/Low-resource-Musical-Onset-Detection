from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch import nn
import torchaudio.transforms as T
from datasets import load_dataset
import numpy as np
from sklearn import metrics
import evaluate
import pickle
from glob import glob
import librosa

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("pretrained_models/MERT-v0", trust_remote_code=True)
sampling_rate = 16000
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

class MertClassifier(torch.nn.Module):
    def __init__(self, model, layer_index):
        super(MertClassifier, self).__init__()
        self.model = model
        # self.feature_extractor = model.feature_extractor
        # self.feature_projection = model.feature_projection
        # self.encoder = model.encoder
        self.layer_index = layer_index
        for i in range(self.layer_index + 1, 12):
            del self.model.encoder.layers[-1]
        self.classifier = nn.Sequential(nn.Linear(in_features=768, out_features=256, bias=True),
                                        nn.Linear(in_features=256, out_features=2, bias=True))

    def forward(self, x):
        # self.model.add_module('classifier', nn.Sequential(nn.Linear(in_features=768, out_features=256, bias=True),
                                                     # nn.Linear(in_features=256, out_features=2, bias=True)))
        # for i in range(self.layer_index + 1, 12):
        #     del self.model.encoder.layers[-1]
        x = self.model(x, output_hidden_states = True)
        x = self.classifier(x.hidden_states[self.layer_index]) # to change the layer index
        # return nn.functional.softmax(x, dim=-1)
        return x

class MertClassifierFreezeGRU(torch.nn.Module):
    def __init__(self, model, layer_index, input_size, hidden_size, output_size):
        super(MertClassifierFreezeGRU, self).__init__()
        self.model = model
        self.layer_index = layer_index
        self

class MertClassifierGRU(torch.nn.Module):
    def __init__(self, model, layer_index, input_size, hidden_size, output_size):
        super(MertClassifierGRU, self).__init__()
        self.model = model
        self.layer_index = layer_index
        for i in range(self.layer_index + 1, 12):
            del self.model.encoder.layers[-1]
        self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(in_features=hidden_size * 2, out_features=output_size, bias=True)

    def forward(self, x):
        # self.model.add_module('classifier', nn.Sequential(nn.Linear(in_features=768, out_features=256, bias=True),
                                                     # nn.Linear(in_features=256, out_features=2, bias=True)))
        # for i in range(self.layer_index + 1, 12):
        #     del self.model.encoder.layers[-1]
        x = self.model(x, output_hidden_states = True)
        x = self.gru(x.hidden_states[self.layer_index]) # to change the layer index
        x = self.classifier(x[0]) # GRU return 2-d outputs
        # return nn.functional.softmax(x, dim=-1)
        return x

class MertClassifierGRUAggre(torch.nn.Module):
    def __init__(self, model, layer_index, input_size, hidden_size, output_size, batch_size, time_step):
        super(MertClassifierGRUAggre, self).__init__()
        self.model = model
        self.input_size = input_size
        self.layer_index = layer_index
        self.batch_size = batch_size
        self.time_step = time_step
        for i in range(self.layer_index + 1, 12):
            del self.model.encoder.layers[-1]
        self.agg = nn.Conv2d(in_channels=layer_index + 2, out_channels=1, kernel_size=1)
        self.gru = nn.GRU(input_size = input_size, hidden_size = hidden_size, batch_first = True, bidirectional = True)
        self.classifier = nn.Linear(in_features = hidden_size * 2, out_features = output_size, bias = True)

    def forward(self, x):
        # self.model.add_module('classifier', nn.Sequential(nn.Linear(in_features=768, out_features=256, bias=True),
                                                     # nn.Linear(in_features=256, out_features=2, bias=True)))
        # for i in range(self.layer_index + 1, 12):
        #     del self.model.encoder.layers[-1]
        x = self.model(x, output_hidden_states = True)

        y = []
        for i in range(min(x.hidden_states[0].shape[0], self.batch_size)): # each hidden state has a batch of 8
            ele = torch.stack([x.hidden_states[j][i] for j in range(self.layer_index + 2)])
            y.append(ele)
        x = torch.stack(y, dim=0)

        x = self.agg(x)
        x = torch.reshape(x, (x.shape[0], self.time_step, self.input_size))
        x = self.gru(x) # to change the layer index
        x = self.classifier(x[0]) # GRU return 2-d outputs
        # return nn.functional.softmax(x, dim=-1)
        return x

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(inputs['input_values'])
        logits = outputs

        # compute custom loss (suppose one has 2 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to(device)) # weighted CE Loss

        # lo = logits.view(-1, self.model.config.num_labels)
        la = labels.view(-1, 2) # num of labels = 2
        indexs = []
        for i in range(len(la)):
            index = torch.argmax(la[i].data)
            indexs.append(index)
            # print(index)
        labels_te = torch.stack(indexs)

        # print(lo)
        # print(la)
        # print(labels_te)
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1, self.model.config.num_labels))

        loss = loss_fct(logits.view(-1, 2), labels_te)
        print(loss)
        return (loss, outputs) if return_outputs else loss

class CustomTrainer_smooth(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(inputs['input_values'])
        logits = outputs

        # compute custom loss (suppose one has 2 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0]).to(device)) # smoothed labels don't need weighted CE Loss at all

        # lo = logits.view(-1, self.model.config.num_labels)
        la = labels.view(-1, 2) # num of labels = 2
        indexs = []
        for i in range(len(la)):
            index = torch.argmax(la[i].data)
            indexs.append(index)
            # print(index)

        labels_te = np.array([index.cpu() for index in indexs])

        # label smoothing
        conv_kernel = np.array([0.1, 0.4, 1, 0.4, 0.1])
        for _ in range(2):
            labels_te = np.convolve(labels_te, conv_kernel, mode='same')
        labels_te = np.clip(labels_te, 0, 1)
        labels_te = torch.tensor(labels_te, dtype=torch.long).to(device)

        # print(lo)
        # print(la)
        # print(labels_te)
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1, self.model.config.num_labels))

        loss = loss_fct(logits.view(-1, 2), labels_te)
        print(loss)
        return (loss, outputs) if return_outputs else loss

def evaluation_metrics(predictions, labels):
    f1_score = metrics.f1_score(labels, predictions, labels=None, pos_label=2,
                             average='weighted', sample_weight=None, zero_division='warn')
    precision = metrics.precision_score(labels, predictions, labels=None, pos_label=2,
                            average='weighted', sample_weight=None, zero_division='warn')
    recall = metrics.recall_score(labels, predictions, labels=None, pos_label=2,
                            average='weighted', sample_weight=None, zero_division='warn')

    print("f1 score: {}, precision: {}, recall: {}".format(f1_score, precision, recall))
    return f1_score, precision, recall

def collate_fn(batch):
    return {
        'input_values': torch.stack([x['input_values'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch]) # tensor 改成 stack 会报错
    }

def compute_metrics(p):
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    p1 = np.argmax(p.predictions, axis=2)
    p2 = np.argmax(p.label_ids, axis=2)
    p1_new = p1.reshape(p1.shape[0] * p1.shape[1])
    p2_new = p2.reshape(p2.shape[0] * p2.shape[1])
    # print(len(p1_new))
    # print(len(p2_new))
    print(clf_metrics.compute(predictions=p1_new, references=p2_new))
    return clf_metrics.compute(predictions=p1_new, references=p2_new)

def finetune_mert(prepared_ds_train, prepared_ds_valid, layer_index, extra, remove_or_not, version): # with trainer
    labels = ["none", "onset"]

    processor = Wav2Vec2FeatureExtractor.from_pretrained("pretrained_models/MERT-v0", trust_remote_code=True)

    model = AutoModel.from_pretrained(
        "pretrained_models/MERT-v0",
        trust_remote_code=True
    )

    # change the layer_index
    # if extra == 'gru':
    #
    myModel = MertClassifier(model, layer_index=layer_index)
    # myModel = MertClassifierGRU(model,
    #                             layer_index=layer_index,
    #                             input_size=768,
    #                             hidden_size=20,
    #                             output_size=2)
    # myModel = MertClassifierGRUAggre(model,
    #                             layer_index=layer_index,
    #                             input_size=768,
    #                             hidden_size=20,
    #                             output_size=2,
    #                             batch_size=8,
    #                             time_step = 249)

    # model = AutoModel.from_pretrained(
    #     "pretrained_models/MERT-v0",
    #     trust_remote_code=True,
    #     num_labels = len(labels),
    #     id2label = {str(i): c for i, c in enumerate(labels)},
    #     label2id = {c: str(i) for i, c in enumerate(labels)}
    # )

    # print(model)
    print(myModel)

    # freeze layers
    # for param in myModel.parameters():
    #     param.requires_grad = False
    # for param in myModel.gru.parameters():
    #     param.requires_grad = True
    # for param in myModel.classifier.parameters():
    #     param.requires_grad = True
    # for param in myModel.model.encoder.layers[5].parameters():
    #     param.requires_grad = True
    # for param in myModel.model.encoder.layers[4].parameters():python 合并array
    #     param.requires_grad = True
    # for param in model.data2vec_audio.encoder.layers[10].parameters():
    #     param.requires_grad = True
    # for param in model.data2vec_audio.encoder.layers[9].parameters():
    #     param.requires_grad = True
    # for param in model.classifier.parameters():
    #     param.requires_grad = True

    trainables = [p for p in myModel.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    training_args = TrainingArguments(
        output_dir="output_models/MERT-v0-finetune/20epochs" + remove_or_not + "-weighted-layer" + str(layer_index) + version + extra,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=20,
        fp16=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
    )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=collate_fn,
    #     compute_metrics=compute_metrics,
    #     train_dataset=prepared_ds_train,
    #     eval_dataset=prepared_ds_valid,
    #     tokenizer=feature_extractor,
    # )

    trainer = CustomTrainer(
        model=myModel,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds_train,
        eval_dataset=prepared_ds_valid,
        tokenizer=processor,
    )

    # train
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # evaluate
    metrics = trainer.evaluate(prepared_ds_valid)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    return

def get_audio_and_label_list(rootpath, time_in_seconds, test_mode=False):
    audio_filepaths = glob(rootpath + "*/*.wav")
    label_filepaths = glob(rootpath + "*/*.txt")
    audio_list = [] # (n, 16000 * 5)
    label_list = [] # (n, 250)
    filename_list = []

    for i in range(len(audio_filepaths)):
        y, sr = librosa.load(audio_filepaths[i], sr=sampling_rate)
        time = librosa.get_duration(y, sr)
        sample_len = time_in_seconds
        hop_len = time_in_seconds
        count = int(time / hop_len) + 1

        # read the onset list of the audio
        onset_list = []
        with open(label_filepaths[i], 'r') as f:
            lines = f.readlines()
            for line in lines:
                onset = float(line.strip().split("\t")[0])
                onset_list.append(onset)

        start_time = 0
        start_onset_index = 0
        end_onset_index = 0

        for j in range(count):
            end_time = start_time + sample_len
            start_point = int(start_time * sr)
            end_point = int(end_time * sr)
            y_j = y[start_point:end_point]

            # padding
            if len(y_j) < sample_len * sr:
                y_pad = np.zeros(sample_len * sr - len(y_j))
                y_j = np.append(y_j, y_pad)

            # get onset of this piece
            for k in range(start_onset_index, len(onset_list)):
                if onset_list[k] > end_time:
                    end_onset_index = k
                    break
                else:
                    end_onset_index += 1 # modify

            label_j = [[1, 0]] * (time_in_seconds * 50 - 1)
            for onset_time in onset_list[start_onset_index:end_onset_index]:
                onset_index = min(int((onset_time - start_time) / 0.02), (time_in_seconds * 50 - 1) - 1)
                label_j[onset_index] = [0, 1]

            start_onset_index = end_onset_index
            start_time += hop_len

            # if end_onset_index < len(onset_list) and not test_mode: # remove the last clip without any onsets
            #     audio_list.append(y_j)
            #     label_list.append(label_j)
            #     filename_list.append(audio_filepaths[i].split("\\")[-1][:-4] + "_" + "%03d" % (j))
            # elif test_mode:
            #     audio_list.append(y_j)
            #     label_list.append(label_j)
            #     filename_list.append(audio_filepaths[i].split("\\")[-1][:-4] + "_" + "%03d" % (j))
            audio_list.append(y_j)
            label_list.append(label_j)
            filename_list.append(audio_filepaths[i].split("\\")[-1][:-4] + "_" + "%03d" % (j))

    return audio_list, label_list, filename_list

def transform_data_to_np(audios_np_file, labels_np_file, pkl_name, time_in_seconds, test_mode=False):
    processed_samples = []
    for i in range(len(audios_np_file)):
        processed_sample = feature_extractor(audios_np_file[i], sampling_rate=sampling_rate, return_tensors="pt")
        if not test_mode:
            processed_sample.data['input_values'] = processed_sample.data['input_values'].reshape(sampling_rate * time_in_seconds)
        processed_sample.data['labels'] = torch.tensor(labels_np_file[i]) # .reshape(1, len(labels_np_file[i]))
        processed_samples.append(processed_sample)

    with open(pkl_name, 'wb') as f1:
        pickle.dump(processed_samples, f1)

    return processed_samples

if __name__ == "__main__":

    # transform the data, perform only once
    # tr_audios_np_file, tr_labels_np_file, tr_filename_list = get_audio_and_label_list("D:/AI_Music/220807-GraduationProject/Datasets/CCOM-HuQin/newset-1-split2-threeSets2/train-augPitch/",
    #                                                                                   time_in_seconds=5,
    #                                                                                   test_mode=False)
    # tr_samples = transform_data_to_np(tr_audios_np_file, tr_labels_np_file, 'mert-v0-data/train_xy_remove.pkl', time_in_seconds=5)
    #
    # vl_audios_np_file, vl_labels_np_file, vl_filename_list = get_audio_and_label_list("D:/AI_Music/220807-GraduationProject/Datasets/CCOM-HuQin/newset-1-split2-threeSets2/valid/",
    #                                                                                   time_in_seconds=5,
    #                                                                                   test_mode=False)
    # vl_samples = transform_data_to_np(vl_audios_np_file, vl_labels_np_file, 'mert-v0-data/valid_xy_remove.pkl', time_in_seconds=5)

    with open('mert-v0-data/train_xy.pkl','rb') as f1:
        tr_samples = pickle.load(f1)
    with open('mert-v0-data/valid_xy.pkl','rb') as f2:
        vl_samples = pickle.load(f2)

    with open('mert-v0-data/train_xy_remove.pkl','rb') as f3:
        tr_samples_rv = pickle.load(f3)
    with open('mert-v0-data/valid_xy_remove.pkl','rb') as f4:
        vl_samples_rv = pickle.load(f4)

    for i in [11]:
        finetune_mert(tr_samples, vl_samples, layer_index=i, extra='', remove_or_not='', version='')
        finetune_mert(tr_samples_rv, vl_samples_rv, layer_index=i, extra='', remove_or_not='remove', version='')