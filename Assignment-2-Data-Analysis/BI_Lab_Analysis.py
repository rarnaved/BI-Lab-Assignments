import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Dataset Loading from CSV
# Make sure your 'dataset.csv' is in the same folder on GitHub
try:
    df = pd.read_csv('dataset.csv')
    print("Dataset loaded successfully!")
    print(df.head())
except Exception as e:
    print("Error loading CSV. Make sure the file exists and has data.")

# 2. Preprocessing (EDA)
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# 3. Topic 5: Performance Evaluation and Comparison
# Features and Target
X = df[['Age', 'Income', 'Credit_Score']]
y = df['Default']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Algorithm 1: Decision Tree ---
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

# --- Algorithm 2: Naive Bayes ---
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)

# 4. Results Comparison
print(f"\n--- Performance Comparison ---")
print(f"Decision Tree Accuracy: {dt_acc * 100:.2f}%")
print(f"Naive Bayes Accuracy: {nb_acc * 100:.2f}%")

# Visualization of Accuracy
methods = ['Decision Tree', 'Naive Bayes']
accuracies = [dt_acc, nb_acc]

plt.figure(figsize=(6,4))
sns.barplot(x=methods, y=accuracies, palette='viridis')
plt.ylim(0, 1)
plt.ylabel('Accuracy Score')
plt.title('Algorithm Performance Comparison')
plt.show()
