#importing all necessary library
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="hibiscus",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# custom styling
st.markdown("""
    <style>
     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        color: #f1f1f1;
    }

    h1, h2, h3 {
        color: #ffffff;
      text-shadow: 1px 1px 3px rgba(0,0,0,0.6);
    }

    .stApp {
        background: linear-gradient(rgba(255,255,255,0.6), rgba(255,255,255,0.6)),
                    url("https://images.unsplash.com/photo-1540163502599-a3284e17072d?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Improve container contrast */
    .css-1d391kg, .css-1cpxqw2, .stButton>button {
        background-color: rgba(0,0,0,0.5);
        color: #f8f8f8;
        border-radius: 10px;
    }

    /* Adjust data table font */
    .dataframe {
        background-color: rgba(0,0,0,0.6);
        color: #f1f1f1;
        border-radius: 8px;
    }

    /* Upload widget styling */
    .css-1umw7bz {
        background-color: rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px;
        color: orange !important;
    }    
    </style>
""", 
unsafe_allow_html=True)
    

#page title
st.title("Iris Flower classification using KNN ML model")
st.write("This app uses the **K-Nearest Neighbour(KNN) algorithm to classify the iris flower based on measurement")

#load Dataset
uploaded_file= st.file_uploader("iris.csv", type=["csv"])
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.write("dataset loaded successfully")
    
    #show first few rows
    st.subheader("📄 First Few Rows of Dataset")
    st.dataframe(df.head())

    #check for required columns
    required_columns={'sepal_length','sepal_width','petal_length','petal_width','species'}
    if not required_columns.issubset(df.columns):
      st.error(f"csv file must include the columns: {required_columns}")
    else:
      # prepare features and labels
      x = df[['sepal_length','sepal_width','petal_length','petal_width']]
      y = df['species']

      #encode species labels into numbers
     # y_encoded = y.astype('category').cat.codes
     # label_mapping = dict(enumerate(y.astype('category').cat.categories))
      y_cat = y.astype('category')
      y_encoded = y_cat.cat.codes
      class_names = list(y_cat.cat.categories)

      # Train_test split
      x_train, x_test, y_train , y_test = train_test_split(
          x, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
      )   

      #feature Scaling
      scaler = StandardScaler()
      x_train_scaled = scaler.fit_transform(x_train)
      x_test_scaled = scaler.transform(x_test)

      # Train the KNN model
      knn_model = KNeighborsClassifier(n_neighbors=3)
      knn_model.fit(x_train_scaled, y_train)

      # Sidebar for user input
      st.sidebar.header("🌼Input Flower Measurements")
      sepal_length = st.sidebar.slider("sepal length(cm)",4.0,8.0,5.1)
      sepal_width = st.sidebar.slider("speal width(cm)",2.0,4.5,3.0)
      petal_length = st.sidebar.slider("petal length(cm)",1.0,7.0,1.4)
      petal_width = st.sidebar.slider("petal width(cm)",0.1,2.5,0.2)

      st.sidebar.subheader("Model settings")
      k = st.sidebar.slider("Select K value for KNN",1, 15, 3)

      
      # Input Display
      user_input = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
      user_input_df = pd.DataFrame(user_input, columns=x.columns)

      st.subheader("user input") 
      st.write(user_input_df) 

      # prediction
      if st.sidebar.button("🔍 Predict"):
       user_input_scaled = scaler.transform(user_input)
       prediction = int(knn_model.predict(user_input_scaled)[0])
       predicted_species = str(class_names[prediction])

       st.subheader("📌 Prediction")
       st.success(f"The predicted species is: **{predicted_species.capitalize()}**")

      # Metrics
       y_pred = knn_model.predict(x_test_scaled)
       accuracy = accuracy_score(y_test, y_pred)
       col1, col2 = st.columns(2)
       with col1:
         st.metric("Accuracy", f"{accuracy*100:.2%}%")
       with col2:
         st.metric("K value used", k)

      # Classification Report        
       st.subheader("📊 Classification Report")
       report = classification_report(y_test, y_pred, target_names=class_names, output_dict= True)
       st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Blues'))

      # confusion Matrix
       st.subheader("🧩 Confusion Matrix")
       fig, ax =plt.subplots()
       cm = confusion_matrix(y_test,y_pred)
       sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=class_names,yticklabels=class_names)
       plt.xlabel("Predicted")
       plt.ylabel("Actual")
       st.pyplot(fig)
    


      # Visualization  
       st.subheader("📊 Data Visualization")
      # if st.checkbox("Pairplot"):
       with st.spinner("Generating Pairplot..."):
           
        plot_df = df.copy()
        fig2 = sns.pairplot(plot_df, hue='species', corner=True)
        st.pyplot(fig2)

       
else:
    st.info("👆 Please upload an Iris dataset CSV file to begin.")   

