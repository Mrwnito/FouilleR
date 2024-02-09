# Chargement des librairies nécessaires
library(tm)
library(SnowballC)
library(e1071)
library(caret)
library(dplyr)

# Chargement des données
setwd('C:\\Users\\HP\\Documents\\Fouilles_donnee_R\\3_mini-projets\\projet_1')
data <- read.csv("Emotion_classify_Data.csv", stringsAsFactors = FALSE)

# Prétraitement des données textuelles
corpus <- VCorpus(VectorSource(data$Comment))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)

# Création de la matrice TF-IDF
dtm <- DocumentTermMatrix(corpus)
tfidf <- weightTfIdf(dtm)
data_tfidf <- as.data.frame(as.matrix(tfidf))
colnames(data_tfidf) <- make.names(colnames(data_tfidf))

# Ajout de la colonne de classe (emotion) aux données
data_tfidf$emotion <- data$Emotion

# Division en données d'entraînement et de test
set.seed(42)
training_rows <- createDataPartition(data_tfidf$emotion, p = 0.8, list = FALSE)
train_data <- data_tfidf[training_rows, ]
test_data <- data_tfidf[-training_rows, ]

# Entraînement du modèle bayésien naïf
model <- naiveBayes(emotion ~ ., data = train_data)

# Prédictions et évaluation du modèle
predictions <- predict(model, test_data)
conf_matrix <- confusionMatrix(predictions, test_data$emotion)

# Affichage des résultats
print(conf_matrix)