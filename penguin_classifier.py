import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 

# Load the penguin dataset
file_path = 'penguins.csv'

df = pd.read_csv('penguins.csv')
print(df.head())

# inputs 
 
x = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]

# outputs 

y = df['species']

# remove NaN 

x = x.dropna() 

y = y[x.index]

# split the data

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=80,test_size=0.2, shuffle=True)

#print the shape of the data

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# model 

model = DecisionTreeClassifier() # empty decision tree blueprint 

model.fit(x_train,y_train)

predicted_species = model.predict(x_test) # predict the species for the test set

# plot 

feature_x = 'bill_length_mm'
feature_y = 'flipper_length_mm'

plot_df = x_test.copy()
plot_df['predicted_species'] = predicted_species
plot_df['true_species'] = y_test

plot_df["correct"] = plot_df["true_species"] == plot_df["predicted_species"]

plt.figure(figsize=(10, 7))

species_list = plot_df["true_species"].unique()
markers = {True: "o", False: "X"}   # circle = correct, X = wrong

for species in species_list:
    for correct in [True, False]:
        subset = plot_df[
            (plot_df["true_species"] == species) &
            (plot_df["correct"] == correct)
        ]

        plt.scatter(
            subset[feature_x],
            subset[feature_y],
            label=f"{species} ({'correct' if correct else 'wrong'})",
            marker=markers[correct],
            alpha=0.7,
            s=80
        )

plt.xlabel("Bill Length (mm)", fontsize=12)
plt.ylabel("Flipper Length (mm)", fontsize=12)
plt.title("Penguin Species Clustering (Decision Tree Predictions)", fontsize=14)

plt.legend()
plt.grid(True)
plt.show()


# accuracy 
accuracy = accuracy_score(y_test,predicted_species)

print("Accuracy:", accuracy)

# confusion matrix

conf_matrix = confusion_matrix(y_test, predicted_species)

print("Confusion Matrix:", conf_matrix)

class_names = model.classes_

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=class_names)
disp.plot()
plt.title("Confusion Matrix for Penguin Species Classification")
plt.show()


