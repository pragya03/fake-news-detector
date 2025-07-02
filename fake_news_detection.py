import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Load data
df_fake = pd.read_csv('Fake.csv')
df_true = pd.read_csv('True.csv')

# Label assignment
df_fake['label'] = 0
df_true['label'] = 1

# Balance the dataset
min_len = min(len(df_fake), len(df_true))
df_fake_balanced = df_fake.sample(min_len, random_state=42)
df_true_balanced = df_true.sample(min_len, random_state=42)

# Combine and shuffle
df = pd.concat([df_fake_balanced, df_true_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)

# Preprocessing
X = df['text']
y = df['label']

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))  # 🔧 Better n-gram support
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))