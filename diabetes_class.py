import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset
diabetes_df = pd.read_csv('dataset/diabetes.csv')

print(diabetes_df.head())
diabetes_df.info()

# Check the distribution of the target variable
sns.countplot(diabetes_df['Outcome'])
plt.show()

# correlation plot
sns.heatmap(diabetes_df.corr(), annot=True)
plt.show()

# Split the dataset into features and target
X = diabetes_df.iloc[:, 0:8].values
y = diabetes_df.iloc[:, 8].values

#feature scaling
sc = StandardScaler() # StandardScaler is used to scale the data because the features have different scales and the standard scaler scales the data to have a mean of 0 and a standard deviation of 1, making the data normally distributed. the reason that standard scaler is used instead of minmax scaler is that the data is normally distributed.
X = sc.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=500, activation='relu', input_shape=(8,)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units=500, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs_hist = model.fit(X_train, y_train, epochs=100, batch_size=50, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5) # if the value is greater than 0.5, it will be true, otherwise false

epochs_hist.history.keys()

# Plot the loss
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.ylabel('Training and Validation Loss')
plt.xlabel('Epoch number')
plt.legend(['Training Loss', 'Validation Loss'])

# Plot the accuracy
plt.plot(epochs_hist.history['accuracy'])
plt.plot(epochs_hist.history['val_accuracy'])
plt.title('Model Accuracy Progress During Training')
plt.ylabel('Training and Validation Accuracy')
plt.xlabel('Epoch number')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.show()

# Confusion matrix for the training set
y_train_pred = model.predict(X_train)
y_train_pred = (y_train_pred > 0.5)
cm = confusion_matrix(y_train, y_train_pred)
sns.heatmap(cm, annot=True)

# Classification report for the training set
print(classification_report(y_train, y_train_pred))

# cponfusion matrix for the test set
cm_test = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_test, annot=True)

# Classification report for the test set
print(classification_report(y_test, y_pred))

