"""# Import Dependencies

## Import dependencies
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
from official.nlp import optimization

"""# Cleaned Dataset Import"""

train_df = pd.read_csv('train/train_task2_cleaned_full.csv')
train_df.fillna('', inplace=True)
train_df.info(show_counts=True)

"""# BERT Input Like Sentences generation"""

train_df['text_a'] = train_df['query']
train_df['text_b'] = train_df['product_title'] + '. ' + train_df['product_brand'] + '. ' + train_df['product_color_name']
train_df = train_df.drop(train_df.columns[[1,3,4,5]], axis=1)
train_df.info()
"""# Factorizing Classes

###The category labels are:
* 0 - exact
* 1 - substitute
* 2 - complement 
* 3 - irrelevant     
"""

seq_len = [len(i.split()) for i in train_df['text_a']]
pd.Series(seq_len).hist(bins = 30, log=True)

count = 0
for x in seq_len:
  count = count + x
print("Avg: "+ str(count/len(seq_len)))
print("Max: "+ str(max(seq_len)))

seq_len = [len(i.split()) for i in train_df['text_b']]
pd.Series(seq_len).hist(bins = 30, log=True)

count = 0
for x in seq_len:
  count = count + x
print("Avg: "+ str(count/len(seq_len)))
print("Max: "+ str(max(seq_len)))

"""# Check factorized classes correspondences"""

train_df['esci_label'].value_counts()

"""# Class weights

Since the train dataset is not homogeneous and the goal of the model is to identify complement products matches, but we don't have very many of those samples to work with, so I would want to have the classifier heavily weight the few examples that are available. I am going to do this by passing Keras weights for each class through a parameter. These will cause the model to "pay more attention" to examples from an under-represented class.

Each class weight is calculated as the frequency of that class multiplied by the total number of element divided by the class number.
"""

NUM_ELEMENTS = len(train_df)
weight_for_0 = (1 / train_df['esci_label'].value_counts()['exact'])*(NUM_ELEMENTS/4.0)
weight_for_1 = (1 / train_df['esci_label'].value_counts()['substitute'])*(NUM_ELEMENTS/4.0)
weight_for_2 = (1 / train_df['esci_label'].value_counts()['complement'])*(NUM_ELEMENTS/4.0)
weight_for_3 = (1 / train_df['esci_label'].value_counts()['irrelevant'])*(NUM_ELEMENTS/4.0)

class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2, 3: weight_for_3}
print(class_weight)

train_df['esci_label_code'] = train_df['esci_label'].map( {'exact':0, 'substitute':1, 'complement':2, 'irrelevant':3})
train_df.head()
"""# Initialize bert base multilingual uncased model

## Load Bert Base Multilingual Uncased
"""

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

"""## Initialize Input Token Identification variable
This argument indicates to the model token indices, numerical representations of tokens building the sequences.

"""

X_input_ids = np.zeros((len(train_df), 160))

"""## Initialize Attention Mask variable
This argument indicates to the model which tokens should be attended to, and which should not.
"""

X_attn_masks = np.zeros((len(train_df), 160))
X_token_ids = np.zeros((len(train_df), 160))

train_df.info()

"""# Fine Tuning

## Function to generate the Bert understandable dataset
"""

def generate_training_data(df, ids, masks, token, tokenizer):
    for i in tqdm(range(0, len(train_df))):
        tokenized_text = tokenizer(
            text=train_df.iloc[i,2],
            text_pair=train_df.iloc[i,3],
            max_length=160,
            truncation= 'only_second',
            padding='max_length',
            add_special_tokens=True,
            return_tensors='tf'
        )
        ids[i, :] = tokenized_text.input_ids
        masks[i, :] = tokenized_text.attention_mask
        token[i, :] = tokenized_text.token_type_ids
    return ids, masks, token

X_input_ids, X_attn_masks,X_token_ids = generate_training_data(train_df, X_input_ids, X_attn_masks,X_token_ids, tokenizer)

X_input_ids.shape

"""## Initialize train labels variable"""

labels = np.zeros((len(train_df), 4),dtype=int)

"""## One Hot encoding"""

labels[np.arange(len(train_df)), train_df['esci_label_code'].values] = 1 # one-hot encoded target tensor

"""## Build Tensorflow input"""

dataset = tf.data.Dataset.from_tensor_slices((X_input_ids, X_attn_masks, X_token_ids, labels))

def ESCIDatasetMapFunction(input_ids, attn_masks,token_ids, labels):
    return {
        'input_ids': input_ids,
        'token_type_ids':token_ids,
        'attention_mask': attn_masks
    }, labels

"""We are using drop_remainder argument to ignore that last batch, and get full shape propagation."""

dataset = dataset.map(ESCIDatasetMapFunction) # converting to required format for tensorflow dataset
dataset = dataset.shuffle(len(train_df)).batch(16, drop_remainder=True)

p = 0.80
train_size = int((len(train_df)//16)*p)
train_size

"""## Dividing train_df into Train and Validation Dataset"""

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

"""## Load pretrained Bert model"""

model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased') # bert base model with pretrained weights

"""## Define model sequential layers

Output layer has softmax activation in order to set initial bias to a value that correctly represents the imbalanced training dataset, some classes has much more instances than others ('exact' = 65% of train dataset)


"""

input_ids = tf.keras.layers.Input(shape=(160,), name='input_ids', dtype='int32')
attn_masks = tf.keras.layers.Input(shape=(160,), name='attention_mask', dtype='int32')
token_ids = tf.keras.layers.Input(shape=(160,), name='token_type_ids', dtype='int32')
#da tf documentazione
bert_embds = model.bert(input_ids=input_ids, attention_mask=attn_masks, token_type_ids=token_ids)[1] # 0 -> activation layer (3D), 1 -> pooled output layer (2D)
#intermediate layer
hidden_layer_1 = tf.keras.layers.Dense(128,activation='relu', name='hidden_layer_1')(bert_embds)
dropout = tf.keras.layers.Dropout(.1)(hidden_layer_1)
output_layer = tf.keras.layers.Dense(4, activation='softmax', name='output_layer')(dropout) # softmax -> calcs probs of classes

esci_model = tf.keras.Model(inputs=[input_ids, attn_masks, token_ids], outputs=output_layer)
esci_model.summary()

"""## Define Hyperparameters"""

# hyperparameters
#optim = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
epochs = 4
steps_per_epoch = 3000
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 5e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')
loss_func = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

"""## Compile the model"""

esci_model.compile(optimizer=optimizer, loss=loss_func, metrics=[acc])

"""## Train the model"""

hist = esci_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    class_weight=class_weight
)

"""## Save the trained model for inference"""

esci_model.save('models/esci_model_mlp_adamw_no_optimizer_8_epochs_200k', include_optimizer=False)

esci_model.save('models/esci_model_mlp_adamw_with_optimizer_8_epochs_200k', include_optimizer=True)

"""# Plot Model Training"""

# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()