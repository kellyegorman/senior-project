import pandas as pd 
import re 
import nltk 
from nltk.corpus import stopwords 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from keras.utils import to_categorical
import tensorflow.keras.backend as K
# from keras import backend as K
import joblib
import matplotlib.pyplot as plt
# from backend.content_summaries.TEST_url_text import get_url_text
# followed tutorial : https://www.kaggle.com/code/victornicofac/hate-speech-and-offensive-language-detection

data = pd.read_csv('backend/content_flagging/labeled_data.csv')
# viewing data
# stopwords
# print(data.head())
# print(data['class'].unique())
# print(data[data['class'] == 0])
#print(f"num of tweets: {data.shape}")
# --> num of tweets: (24783, 7)

# extract the text and labels
tweet = list(data['tweet'])
labels = list(data['class'])

# functions to clean data (source: Kaggle)
stop_words = set(stopwords.words('english'))
# add rt to remove retweet in dataset (noise)
stop_words.add("rt")

# remove html entity:
def remove_entity(raw_text):
    entity_regex = r"&[^\s;]+;"
    text = re.sub(entity_regex, "", raw_text)
    return text

# change the user tags
def change_user(raw_text):
    regex = r"@([^ ]+)"
    text = re.sub(regex, "user", raw_text)

    return text

# remove urls
def remove_url(raw_text):
    url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text = re.sub(url_regex, '', raw_text)

    return text

# remove unnecessary symbols
def remove_noise_symbols(raw_text):
    text = raw_text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("!", '')
    text = text.replace("`", '')
    text = text.replace("..", '')

    return text

# remove stopwords
def remove_stopwords(raw_text):
    tokenize = nltk.word_tokenize(raw_text)
    text = [word for word in tokenize if not word.lower() in stop_words]
    text = " ".join(text)

    return text

## this function in to clean all the dataset by utilizing all the function above
def preprocess(datas):
    clean = []
    # change the @xxx into "user"
    clean = [change_user(text) for text in datas]
    # remove emojis (specifically unicode emojis)
    clean = [remove_entity(text) for text in clean]
    # remove urls
    clean = [remove_url(text) for text in clean]
    # remove trailing stuff
    clean = [remove_noise_symbols(text) for text in clean]
    # remove stopwords
    clean = [remove_stopwords(text) for text in clean]

    return clean

def get_url_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        page_text = soup.get_text(separator='\n', strip=True)
        return page_text
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# split train/validation
X_train, X_test, y_train, y_test = train_test_split(clean_tweet, labels, test_size=0.2, random_state=42)
# tokenize
tokenizer = Tokenizer()
# build the vocabulary based on train dataset
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
# vocabulary size (num of unique words) -> will be used in embedding layer
vocab_size = len(tokenizer.word_index) + 1
joblib.dump(tokenizer,'backend/content_flagging/tokenizer.sav')
joblib.load('backend/content_flagging/tokenizer.sav')

#padding
max_length = max(len(seq) for seq in X_train)
joblib.dump(max_length,'backend/content_flagging/max_length.sav')
# test outlier case 
for x in X_test:
    if len(x) > max_length:
        print(f"an outlier detected: {x}")
# uniformize sequences
X_train = pad_sequences(X_train, maxlen = max_length)
X_test = pad_sequences(X_test, maxlen = max_length)

# create hot_labels
y_test = to_categorical(y_test, num_classes=3)
y_train = to_categorical(y_train, num_classes=3)

print(f"num test tweet: {y_test.shape[0]}")
print(f"num train tweet: {y_train.shape[0]}")


# functions to build the model
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precisions = precision(y_true, y_pred)
    recalls = recall(y_true, y_pred)
    return 2*((precisions*recalls)/(precisions+recalls+K.epsilon()))

# adjust + Test
output_dim = 200

# LSTM model architechture (CNN + LSTM)
model = Sequential([
    Embedding(vocab_size, output_dim, input_length=max_length),
    # lstm for xxx
    LSTM(64, dropout=0.3, recurrent_dropout=0.3),
    # dropout to prevent overfitting
    Dropout(0.5),
    # dense to connect the previous output with current layer
    Dense(128, activation="relu"),
    # dropout to prevent overfitting
    Dropout(0.5),
    # this is output layer, with 3 class (0, 1, 2)
    Dense(3, activation="softmax"),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',f1,precision, recall])
# check model parameters
# model.summary() <-- check this again later looked kind of weird??

# Train the model
model_history = model.fit(
    X_train,
    y_train,
    batch_size = 64,
    epochs=10,
    validation_data=(X_test, y_test)
)

# save model
model.save("backend/content_flagging/offensive_model.keras")

hist = model.history.history
plt.plot(hist['loss'],'r',linewidth=2, label='Training loss')
plt.plot(hist['val_loss'], 'g',linewidth=2, label='Validation loss')
plt.title('Hate Speech and Offensive language Model')
plt.xlabel('Epochs numbers')
plt.ylabel('MSE numbers')
plt.show()


# XXXXX TEST w/ URL TEXT

# def predict_offensive(text):
#     clean_text = preprocess(text)
#     seq = tokenizer.texts_to_sequences([clean_text])
#     padded_seq = pad_sequences(seq, maxlen=max_length)
#     pred = model.predict(padded_seq)
#     pred_label = pred.argmax(axis=1)[0]
#     return pred_label

# def predict_offensive_from_url(url):
#     url_text = get_url_text(url)
#     if url_text:
#         return predict_offensive(url_text)
#     else:
#         return None
    
# if __name__ == "__main__":
#     print(predict_offensive_from_url("https://www.reddit.com/r/AmItheAsshole/comments/15h8l7u/aita_for_telling_my_sister_to_stop_talking_to/"))