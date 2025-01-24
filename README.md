# SmartEater
1. Initial data loading and preprocessing

The initial loading was done in colab notebook Preprocessing.ipynb. The embeddings of text reviews were created in the .py files. The output of this step is the following: reviews embeddings performed with both word2vec and fasttext, and philadelphia_restaurants.csv. 

Due to the size of initial JSON datasets, and the lot of computation involved in processing them, we recommend skipping this stage. Additionally, some files were renamed, so executing colab notebook and .py scripts will give content-wise correct files, but not naming-wise. 

2. restaurants and users data preprocessing

User data preprocessing:

Please ensure that the path to the file is correct.

In the folder named “2. restaurants and users data preprocessing,” the file user_processing.py is responsible for preprocessing the user data set. It takes a JSON file yelp_academic_dataset_user.json, which is the original dataset from Yelp as an input; it outputs the file “User_csv_with_embeddings.csv.” 
To run the user_processing.py file, please download: yelp_academic_dataset_user.json and install all necessary dependencies - pandas, scikit-learn.

Restaurant data preprocessing:

Please ensure that the path to the file for the .py script is correct. In the colab notebook, we use the direct upload function. Please ensure that the path to the file is correct.

In the folder named “2. restaurants and users data preprocessing,” 2 files are responsible for preprocessing the restaurant data set. 

The script create_restaurant_final_embeddings.py processes raw restaurant data to create a dataset with meaningful embeddings for machine learning and analysis. It begins with a dataset of Philadelphia restaurants (philadelphia_restaurants.csv), where categorical features are transformed into compact representations to reduce dimensionality, such as combining one-hot-encoded category columns into a single categories column. Columns with more than 50% missing values are dropped, and ordinal encoding is applied to features like noise level, WiFi, and attire. True/False columns are standardized, missing values are imputed, and numerical columns like price range are analyzed for skewness to determine the most suitable imputation method. The processed data is saved as out.csv.

In the next stage, use create_restaurant_final_embeddings_complete.ipynb to preprocess data further. Preprocessing includes normalizing numerical features like latitude, longitude, and review count using MinMax scaling. Pre-trained word embeddings from the GloVe model are used to convert the categories column into vector representations, enabling semantic understanding of restaurant categories. Finally, embeddings for other selected features, including numerical, categorical, and true/false attributes, are concatenated with category embeddings to form a comprehensive final_embedding for each restaurant. The output is saved as philadelphia_restaurants_with_embeddings.csv.

3. DNN

In both of the following colab notebooks, we mount a drive. Please download the necessary .csv or .parquet files and provide a correct drive path to load input files correctly. 

EmbeddingProcessing_final.ipynb is designed to prepare embeddings for the DNN model. It uses files that we prepared in steps 1 and 2. The input files are philadelphia_restaurants_with_embeddings.csv, User_csv_with_embeddings.csv, and philadelphia_reviews_with_fasttext_embeddings.parquet. This colab notebook takes embeddings of restaurants and users, and adds to them reviews embeddings. For restaurants, it’s the average of reviews they receive. For users, it’s the average of reviews this user posted. The output is 2 files - restaurants_with_pca_embeddings.csv and users_with_pca_embeddings.csv. These files are of the same dimensions and will be used as input for the model. 

ModelRecommend_final.ipynb is a colab notebook that actually contains a model. The files that are used: restaurants_with_pca_embeddings.csv, users_with_pca_embeddings.csv, philadelphia_reviews_with_fasttext_embeddings.parquet. 

The function recommend_restaurants(user_id, k, review_df, restaurant_df, model) is used to produce recommendations. Inputs that can be controlled are user_id and k, where user_id is the id of a user from datasets and k is the number of recommended restaurants to output. In the notebook, there is a sample user_id and k set to 5. 


4. Category-based model

The model uses the parquet file ‘philadelphia_reviews_with_word2vec_embeddings.parquet’ and outputs a CSV file ‘sentiment_analysis_results.csv’. Each row in the output contains the business_id, review count, and a score for each sentiment category. It first embeds the category templates using a pre-trained FastText model. It then compares these category template embeddings to the embedding for each review (contained in the input file) to generate the sentiment scores. The model takes some time to run so the output used in the analysis is provided in the drive (‘sentiment_analysis_results.csv’). 
