### IDENTIFY HUMAN VALUES BEHIND THE ARGUMENT ###
#librerie utilizzate
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer 
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import numpy as np
#DATASET 
arguments_training = pd.read_csv("\Dataset\arguments-training.tsv", delimiter='\t', index_col=False)
arguments_test = pd.read_csv("\Dataset\arguments-test.tsv", delimiter='\t', index_col=False)
arguments_validation = pd.read_csv("\Dataset\arguments-validation.tsv", delimiter='\t', index_col=False)
arguments_test_nahjalbalagha = pd.read_csv("\Dataset\arguments-test-nahjalbalagha.tsv", delimiter='\t', index_col=False)
arguments_validation_zhihu = pd.read_csv("\Dataset\arguments-validation-zhihu.tsv", delimiter='\t', index_col=False)

labels_training = pd.read_csv("\Dataset\labels-training.tsv", delimiter='\t', index_col=False)
labels_test = pd.read_csv("\Dataset\labels-test.tsv", delimiter='\t', index_col=False)
labels_validation = pd.read_csv("\Dataset\labels-validation.tsv", delimiter='\t', index_col=False) 
labels_test_nahjalbalagha = pd.read_csv("\Dataset\labels-test-nahjalbalagha.tsv", delimiter='\t', index_col=False)
labels_validation_zhihu = pd.read_csv("\Dataset\labels-validation-zhihu.tsv", delimiter='\t', index_col=False)

df_arguments=pd.concat([arguments_training,arguments_test,arguments_validation,arguments_test_nahjalbalagha,arguments_validation_zhihu])
df_labels=pd.concat([labels_training,labels_test,labels_validation,labels_test_nahjalbalagha,labels_validation_zhihu])

#drop for debug
#col=['Conclusion','Stance','Premise']
a = df_labels.drop('Argument ID', axis=1)

#1)
#Y = [human_value1, human_value2, ... , human_value20]
#human_value_encoded = [1 ... 20] corrispondente ai valori umani
Y=a.values
human_values = df_labels.columns[1:]


#tokenizare con gli attributi speciali per l'arguments in modo che bert individui correlazione tra premessa e conclusione
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", down_lower_case=True)

MAX_LEN = 60
#1 token creation
conclusion_input_ids = [tokenizer.encode(sent, add_special_tokens=True, max_length=MAX_LEN, pad_to_max_length=True, truncation=True) for sent in df_arguments['Conclusion']]
stance_tokens = df_arguments['Stance'].apply(lambda x: tokenizer.encode(f"[STANCE: {x}]", max_length=7, pad_to_max_length=True, add_special_tokens=False)).tolist()
premise_input_ids = [tokenizer.encode(sent, add_special_tokens=True, max_length=MAX_LEN, pad_to_max_length=True, truncation=True) for sent in df_arguments['Premise']]

conclusion_attention_masks = [[float(i>0) for i in seq] for seq in conclusion_input_ids]
premise_attention_masks = [[float(i>0) for i in seq] for seq in premise_input_ids]
stance_attention_masks = [[float(i > 0) for i in seq] for seq in stance_tokens]

input_ids=[row1+row2+row3 for row1,row2,row3 in zip(conclusion_input_ids,stance_tokens,premise_input_ids)]
attention_masks= [row1+row2+row3 for row1,row2,row3 in zip(conclusion_attention_masks,stance_attention_masks,premise_attention_masks)]

#2 dividing set for training
train_inputs, validation_inputs, train_masks, validation_masks, train_labels, validation_labels = train_test_split(input_ids,attention_masks, Y, random_state=42, test_size=0.2)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
validation_labels = torch.tensor(validation_labels, dtype=torch.float32)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

#3 DataLoader for training and test dataset to iterate over batches of data
#batches dimension = 16 to not overload GPU
batch_size = 16
train_data = TensorDataset(train_inputs,train_masks,train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data,sampler=train_sampler,batch_size=batch_size)

validation_data = TensorDataset(validation_inputs,validation_masks,validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data,sampler=validation_sampler,batch_size=batch_size)

#BERT Model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=20)
#hyper parameter : :
lr= 2e-5
adam_epsilon = 1e-8
epochs=15
num_warmup_steps=0
num_training_steps = len(train_dataloader)*epochs
#instate optimizer and schedules :
optimizer = AdamW(model.parameters(), lr=lr,eps=adam_epsilon,correct_bias=False) 
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

#CUDA for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)
SEED = 19
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device == torch.device("cuda"):
    torch.cuda.manual_seed_all(SEED)
model.to(device)

## Store our loss and accuracy for plotting
train_loss_set = []
learning_rate = []
# Gradients gets accumulated by default
model.zero_grad()
loss_function = torch.nn.BCEWithLogitsLoss()
df_global_metrics = pd.DataFrame(columns=['Epoch', 'Global Accuracy', 'Global Precision', 'Global Recall', 'Global F1'])
df_human_metrics = pd.DataFrame(columns=['Epoch', 'Human Value', 'Accuracy', 'Precision', 'Recall', 'F1'])

#training phase
model.train()  # set model in training phase
for epoch in range(epochs):
    label_predictions = {}
    for i, value in enumerate(human_values):
        label_predictions[value] = {'labels': [], 'predictions': []}
    batch_loss = 0
    for batch_inputs, batch_masks, batch_labels in train_dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_masks = batch_masks.to(device)
        batch_labels = batch_labels.to(device)

        # forward pass
        outputs = model(batch_inputs, attention_mask=batch_masks, labels=batch_labels)
        logits = outputs.logits
        # calculate loss
        loss = loss_function(logits, batch_labels.float())

        # gradient and weight update
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        optimizer.zero_grad()
        batch_loss += loss.item()

    avg_train_loss = batch_loss / len(train_dataloader)
     # store the current learning rate
    for param_group in optimizer.param_groups:
        print("\n\tCurrent Learning rate: ", param_group['lr'])
        learning_rate.append(param_group['lr'])

    train_loss_set.append(avg_train_loss)
    print(F'\n\tAverage Training loss: {avg_train_loss}')
        
# Put model in evaluation mode to evaluate loss on the validation set
    model.eval()
    for batch in validation_dataloader :
    # Add batch to GPU
    # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_input_mask = b_input_mask.to(device)
        b_labels = b_labels.to(device)
    # Telling the model not to compute or store gradients
        with torch.no_grad():
      # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        # Move logits and labels to CPU
        logits = logits[0].to('cpu')
        label_ids = b_labels.to('cpu')
        #Risultato sul set di test
        #Strato finale di decisione sui risultati tramite una sigmoid function
        sigmoid = torch.nn.Sigmoid()
        predictions = torch.sigmoid(logits)
        b_predictions = (predictions > 0.5).int()

        predictions_array = b_predictions.numpy()
        labels_array=label_ids.numpy()

        for i, value in enumerate(human_values):
            label_predictions[value]['labels'].extend(labels_array[:, i].tolist())
            label_predictions[value]['predictions'].extend(predictions_array[:, i].tolist())
    #evaluation metrics, global and specific for class
    global_accuracy=accuracy_score(labels_array.reshape(-1),predictions_array.reshape(-1))
    global_precision=precision_score(labels_array.reshape(-1),predictions_array.reshape(-1), average='weighted')
    global_recall=recall_score(labels_array.reshape(-1),predictions_array.reshape(-1), average='weighted')
    global_f1=f1_score(labels_array.reshape(-1),predictions_array.reshape(-1), average='weighted')
    global_row={'Epoch':epoch,'Global Accuracy':global_accuracy, 'Global Precision':global_precision, 'Global Recall':global_recall, 'Global F1':global_f1}
    df_global_metrics=df_global_metrics.append(global_row,ignore_index=True)
    print(f"Global value")
    print(f"Epoch: {epoch}")
    print(f"Accuracy: {global_accuracy:.4f}")
    print(f"Precision: {global_precision:.4f}")
    print(f"Recall: {global_recall:.4f}")
    print(f"F1-score: {global_f1:.4f}")
    print()


    for value in label_predictions: 
        labels = label_predictions[value]['labels']
        predictions = label_predictions[value]['predictions']

        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        hv_row={'Epoch':epoch, 'Human Value':value, 'Accuracy':accuracy, 'Precision':precision, 'Recall':recall, 'F1':f1}
        df_human_metrics=df_human_metrics.append(hv_row,ignore_index=True)
        print(f"Epoch: {epoch}")
        print(f"Value: {value}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print()

pt_save_directory = "\Model"
tokenizer.save_pretrained(pt_save_directory)
model.save_pretrained(pt_save_directory)
df_human_metrics.to_csv('df_human_metrics.csv', index=False)
df_global_metrics.to_csv('global_metrics.csv',index=False)  
#questa porzione di codice è stata utilizzata in un altro foglio python per visualizzare risultati con le metriche salvate
#riporto qui i grafici utilizzati per la visualizzazione delle metriche
label_counts = df_labels.iloc[:, 1:].sum()

# label count
for label, count in label_counts.items():
    print(f"Label: {label}, Count: {count}")
result= pd.read_csv(r"C:\Users\39371\Desktop\UUU\HumanValue\HumanValue\HumanValue\df_human_metrics.csv", delimiter=',', index_col=False)
result1=result[result['Epoch']==14]
num_campioni = [1394,2306,464,307,2609,888,885,646,3392,2652,936,1953,330,621,2339,1258,3395,708,1090,1943]
f1_score = result1['Precision'].tolist()
labels = result1['Human Value'].tolist()
precision_values=result1['Precision'].tolist()
# Conversione delle precisioni in radianti

precision_rad = [p * 2 * np.pi for p in precision_values]
#GRAFICO PRECISION SCORE PER CLASS
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
ax.plot(angles + angles[:1], precision_rad + precision_rad[:1], linewidth=2, color='cyan')
ax.set_xticks(angles)
ax.set_xticklabels(labels)
ax.scatter(angles, precision_rad, color='cyan', s=50)
for i, label in enumerate(labels):l
    angle_rad = angles[i]
    precision_rad_val = precision_rad[i]
    x = angle_rad
    y = precision_rad_val + 0.1  # Sposta leggermente l'etichetta sopra il punto
    plt.text(x, y, f'{precision_values[i]:.2f}', ha='center', va='center')

#2
# Visualizzazione del grafico
plt.show()
plt.scatter(num_campioni, f1_score)
plt.xlabel('Samples per Class')
plt.ylabel('Precision')
plt.title('Correlation between number of samples and F1')
# Aggiunta delle etichette delle classi
for i, label in enumerate(labels):
    plt.annotate(label, (num_campioni[i], f1_score[i]), textcoords="offset points", xytext=(0,10), ha='center')
# Visualizzazione del grafico
plt.grid(True)
plt.show()

#3
fig, ax1 = plt.subplots()
ax1.bar(result1['Human Value'], result1['F1'], color='cyan')
ax1.set_ylabel('F1')
plt.xticks(rotation=70, fontsize=8)
ax2 = ax1.twinx()
ax2.plot(result1['Human Value'], result1['F1'], marker='o', color='blue')
ax2.set_ylabel('F1', color='red')
# Imposta i limiti dell'asse y tra 0 e 1
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 1)
ax1.set_xlabel('Human Value')
plt.title('F1 for human value')
plt.show()
