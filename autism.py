import streamlit as st
import pandas as pd

# Load dataset
df = pd.read_csv("D:/TEKWORKS/day17/Autism_Data.arff")

# -------------------------------
# DATA PREPROCESSING
# -------------------------------

# Clean age
df["age"] = df["age"].replace("?", 0).astype(int)

# Convert categorical to numeric
df["gender"] = df["gender"].replace({"m": 1, "f": 0})
df["jundice"] = df["jundice"].replace({"yes": 1, "no": 0})
df["austim"] = df["austim"].replace({"yes": 1, "no": 0})
df["age_desc"] = df["age_desc"].replace({"'18 and more'": 18}).astype(int)

# -------------------------------
# ONE HOT ENCODING (RELATION)
# -------------------------------
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False)
relation_encoded = ohe.fit_transform(df[["relation"]])

relation_cols = ohe.get_feature_names_out(["relation"]).tolist()

relation_df = pd.DataFrame(relation_encoded, columns=relation_cols).astype(int)

df = pd.concat([df, relation_df], axis=1)
df.drop("relation", axis=1, inplace=True)

# -------------------------------
# FEATURE SELECTION (DYNAMIC)
# -------------------------------
selected_features = ["age", "gender", "jundice", "austim", "age_desc"] + relation_cols

X = df[selected_features]
y = df["Class/ASD"]

# -------------------------------
# MODEL TRAINING
# -------------------------------
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

model = SVC()
model.fit(X_train, y_train)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("Autism Prediction App")

age = st.number_input("Enter age", 0, 100, 25)
gender = st.selectbox("Select gender", ["m", "f"])
jundice = st.selectbox("Does the person have jundice?", ["yes", "no"])
austim = st.selectbox("Any autism in family?", ["yes", "no"])


# Dynamic relation options (from dataset)
relation_options = [col.replace("relation_", "").replace("_", " ") for col in relation_cols]
relation = st.selectbox("Select relation", relation_options)


# age description is derived from age, so we can calculate it dynamically
if age < 18:
    age_desc = age
else:
    age_desc = 18


# -------------------------------
# CREATE INPUT DATA (DYNAMIC)
# -------------------------------
input_data = pd.DataFrame({
    "age": [age],
    "gender": [1 if gender == "m" else 0],
    "jundice": [1 if jundice == "yes" else 0],
    "austim": [1 if austim == "yes" else 0],
    "age_desc": age_desc
})

# Add relation columns dynamically
for col in relation_cols:
    formatted_col = col.replace("relation_", "").replace("_", " ")
    input_data[col] = 1 if formatted_col == relation else 0

# Ensure exact column match
input_data = input_data.reindex(columns=selected_features, fill_value=0)

# -------------------------------
# PREDICTION
# -------------------------------
prediction = model.predict(input_data)

if prediction[0] == 1:
    st.write("The person is likely to have autism.")
else:
    st.write("The person is unlikely to have autism.")