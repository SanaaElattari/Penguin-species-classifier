# Penguin-species-ml-classifier

A simple machine learning project that predicts penguin species using physical measurements such as bill length, bill depth, flipper length, and body mass.

This project uses a **Decision Tree Classifier** from Scikit-Learn to classify penguins into three species:
- Adelie
- Chinstrap
- Gentoo

The goal of the project is to demonstrate the end-to-end workflow of a machine learning classification problem including data preprocessing, training, prediction, and visualization.

---

## Dataset

The model uses the **Palmer Penguins dataset**, which contains biological measurements of penguins.

Features used in the model:
- Bill Length (mm)
- Bill Depth (mm)
- Flipper Length (mm)
- Body Mass (g)

Target variable:
- Penguin species

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib

---

## Machine Learning Workflow

The model follows these steps:

1. Load and explore the dataset
2. Select relevant numerical features
3. Handle missing values
4. Split the dataset into training and test sets
5. Train a **Decision Tree Classifier**
6. Predict species on unseen data
7. Evaluate performance using:
   - Accuracy score
   - Confusion matrix
8. Visualize predictions with feature scatter plots

---

## Model Performance

The model is evaluated using:

- **Accuracy Score**
- **Confusion Matrix**

The confusion matrix helps visualize how well the model distinguishes between penguin species.

Example confusion matrix output:

| True \ Predicted | Adelie | Chinstrap | Gentoo |
|------------------|--------|-----------|--------|
| Adelie | 39 | 3 | 0 |
| Chinstrap | 2 | 8 | 0 |
| Gentoo | 0 | 0 | 17 |

---

## Visualization

The project includes visualizations to better understand model performance:

- Feature scatter plot comparing **bill length vs flipper length**
- Markers show **correct vs incorrect predictions**
- Confusion matrix heatmap

These plots help illustrate how well the model separates different species.

---

## How to Run the Project

1. Clone the repository

```bash
git clone https://github.com/yourusername/PenguinML_Predictor.git

2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib

3. Run the Program
```bash
python penguin_classifier.py

