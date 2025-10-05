# DHC-3rd-Week
Task 1: Text Classification with BERT on AG News

  This task demonstrates text classification using the AG News dataset. It uses the transformers library with BERT (bert-base-      uncased) for sequence classification.
  Steps include:

  Installing dependencies (transformers, datasets, torch, scikit-learn, gradio).

  Loading and tokenizing the AG News dataset.

  Building and training a BERT model using the Hugging Face Trainer API.

  Evaluating model performance using accuracy and F1-score.

  Deploying the model with a Gradio web interface for real-time text classification.

Task 2: End-to-End Customer Churn Prediction Pipeline

  This task builds a machine learning pipeline to predict customer churn using the Telco Churn dataset. It demonstrates how to      create a structured ML pipeline with scikit-learn.
  Steps include:

  Data loading, preprocessing, and feature encoding.

  Train/test split and application of StandardScaler and LabelEncoder.

  Model training with hyperparameter tuning using GridSearchCV.

  Evaluating models with accuracy, confusion matrix, and visualization.

  Saving and loading trained models with joblib.

Task 3: Multimodal ML – Housing Price Prediction with Images + Tabular Data

  This task combines tabular data and image features to predict housing prices (multimodal machine learning).
  Steps include:

  Loading house price data (structured + images).

  Preprocessing tabular features with scikit-learn (StandardScaler).

  Extracting image features using a CNN (TensorFlow/Keras).

  Merging tabular and image embeddings for prediction.

  Training a hybrid model and evaluating performance with MAE (Mean Absolute Error).

  Visualizing predictions vs. actual prices.
