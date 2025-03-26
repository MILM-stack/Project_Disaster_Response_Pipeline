### Project: Disaster Response Pipeline

## Table of contents
1. [Project Motivation](#motivation)
2. [Installation](#installation)
3. [Instructions](#instructions)
4. [Screenshot](#screeshots)
5. [File Descriptions](#files)
6. [Results](#results)
7. [Licensing, Authors, and Acknowledgements](#licensing)

## 1. Project Motivation <a name="motivation"></a>

The goal of this project is to create a model that will help disaster response efforts. By analyzing real disaster messages, the model classifies these messages towards multiple categories, enabling better communication with disaster relief agencies. The project provides a final app, where the user is able to input new messages and have them directly classified. 

## 2. Installation <a name="installation"></a>

In order to run this project, you will need 
- Anaconda
- Python version 3.*
- the following libraries:
  - re
  - pickle
  - sys
  - the nltk library & it´s modules (specified in train_classifier.py)
  - numpy
  - pandas
  - sqlalchemy
  - scikit-learn modules (specified in train_classifier.py)
  - Flask
  - plotly

## 3. Instructions <a name="instructions"></a>

1. Clone the following repository to your local machine: https://github.com/MILM-stack/Project_Disaster_Response_Pipeline.git
2. Run the ETL pipeline (this pipeline loads, cleans and stores the data into a database) using the following command:
  2.1. python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponseProject.db
3. Run the NLP pipeline (which loads the df from the database, sets the parameters X, y and categories, tokenizes the messages, creates & evalutates the model and finally saves it to a pickle file) using the following command:
   3.1. python models/train_classifier.py data/DisasterResponseProject.db models/classifier.pkl
4. Then go to `app` directory: `cd app`
5. Run your web app: `python run.py`
6. Click the `PREVIEW` button to open the homepage or go to http://0.0.0.0:3000/
7. Browse the app to see the data and how the model classification works!

## 4. Screeshots <a name="Screenshots"></a>

1. Landing Page:
   ![image](https://github.com/user-attachments/assets/d340d0c2-6216-43af-b545-5a42ffb83916)

2. Visualizations:
   2.1: Distribution of Message Genres
   ![image](https://github.com/user-attachments/assets/8fe05d5a-72b5-4c6f-bcb8-5fbc3e754282)
   2.2: Distribution of Messages by Category
   ![image](https://github.com/user-attachments/assets/15ccb843-c71f-421c-a356-36b7c22459a7)
   2.3. Top 10 Message Categories
   ![image](https://github.com/user-attachments/assets/28ad3918-e7e5-454a-a234-c6599c5e9594)

3. Message Classification Input
   ![image](https://github.com/user-attachments/assets/36bf72ea-72d2-4bf2-a1f5-cf23fc8f1657)
   ![image](https://github.com/user-attachments/assets/97655a68-fcfc-4f21-ac20-fbaa41ba599f)


## 5. Files Descriptions <a name="File Descriptions"></a>
Here is an overview of the provided files: 

- Workspace_NoteBooks

| - ETL Pipeline Preparation.ipynb #loads, cleans and stores the data into a database

| - ML Pipeline Preparation.ipynb # loads data, sets parameters X, y and categories, tokenizes the messages, creates & evalutates the model & saves to a pickle file

- app

| - template

  |- master.html  # main page of web app

  |- go.html  # classification result page of web app

|- run.py  # Flask file that runs app

- data

|- disaster_categories.csv  # data to process 

|- disaster_messages.csv  # data to process

|- process_data.py

|- DisasterResponseProject.db   # database to save clean data to

- models

|- train_classifier.py 

|- ones you run train_classifier, you will create a classifier.pkl file with a saved model

- README.md

## 6. Results <a name="results"></a>
The results as shown in the Flask web app, which is shown in the screenshots above. 
The user is provided an overview of the distribution of the messages, which is displayed through visualizations. 
Furthermore, there is an message classification input that can be used to give in new messages and have the classified to the appropiate category. 

Based on the F1 Score results, we can also see that there is room for improvement of the model. Further details can be found in the ML Pipeline Preparation.ipynb

## 7. Licensing, Authors, and Acknowledgements <a name="licensing"></a>

Thank you [Appen](https://www.appen.com/) for providing the datasets. The datasets used in this project are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). This allows for the use of the data for non-commercial purposes, provided that appropriate credit is given, a link to the license is included, and any changes made are indicated.

Thank you [Udacity](https://www.udacity.com/) for training materials, the fantastic support from the mentors on the Knowledge Base, as well as Udacity´s AI. All tools provided by Udacity were extremely helpful. 
