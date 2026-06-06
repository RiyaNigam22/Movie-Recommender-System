# 🎬 Movie Recommendation System

A Machine Learning-powered movie recommendation system that suggests similar movies based on content similarity. The project uses Natural Language Processing (NLP) techniques and cosine similarity to provide personalized movie recommendations through an interactive Streamlit web application.

---

## 🚀 Live Demo

https://movie-recommender-system-n265fnkczssroguruhqdhu.streamlit.app/

---

## 📌 Project Overview

Finding the right movie from thousands of available options can be overwhelming. This project addresses that problem by building a recommendation engine that suggests movies similar to a user's selected movie.

The recommendation system analyzes movie metadata such as:

* Genres
* Keywords
* Cast
* Crew
* Overview

These features are processed and converted into numerical vectors, allowing the model to identify movies with similar characteristics.

---

## ✨ Features

* Interactive web interface built with Streamlit
* Content-based movie recommendation engine
* Fast recommendation generation using precomputed similarity matrix
* User-friendly movie selection dropdown
* Instant display of top recommended movies

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Pickle
* Streamlit

### Machine Learning Techniques

* Text Vectorization
* NLP Preprocessing
* Cosine Similarity
* Content-Based Filtering

---

## 📊 Dataset

The project uses movie metadata containing information such as:

* Movie Title
* Genres
* Cast
* Crew
* Keywords
* Overview

The dataset is cleaned and transformed before training the recommendation model.

---

## ⚙️ Project Workflow

### 1. Data Collection

Movie metadata is collected and loaded into a Pandas DataFrame.

### 2. Data Preprocessing

Relevant columns are selected and cleaned.

### 3. Feature Engineering

Important textual features are combined into a single feature representation.

### 4. Vectorization

Text data is converted into numerical vectors using NLP techniques.

### 5. Similarity Calculation

Cosine similarity is calculated between all movies.

### 6. Model Serialization

The processed data and similarity matrix are stored using Pickle:

* model.pkl
* similarity.pkl

### 7. Web Application

A Streamlit application loads the serialized files and generates recommendations in real time.

---

## 📁 Project Structure

```text
Project - Real_Time_Prediction_Model
│
├── app.py
├── model.pkl
├── similarity.pkl
├── requirements.txt
├── README.md
│
├── notebook
│   └── Real_Time_Prediction_file.ipynb
│
├── data
│
└── report
```

---

## ▶️ Running the Project Locally

### Clone Repository

https://github.com/RiyaNigam22/Movie-Recommender-System

### Move Into Project Folder

```bash
cd movie-recommendation-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Application

https://movie-recommender-system-n265fnkczssroguruhqdhu.streamlit.app/

---

## 🎯 Future Improvements

* Movie poster integration using TMDB API
* Search functionality
* Genre-based filtering
* User authentication
* Hybrid recommendation system
* Recommendation confidence scores
* Responsive modern UI

---

## 👩‍💻 Author

**Riya Nigam**

B.Tech CSE (Business Systems Engineering)

Passionate about Machine Learning, Data Analytics, Product Building, and AI-powered solutions.

---

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub.
