knitr::opts_chunk$set(echo = TRUE)

rm(list = ls())

# Charger les bibliothèques nécessaires
library(tidytext)
library(dplyr)
library(tm)
library(e1071)
library(caret)

# Charger les données
setwd('C:\\Users\\HP\\Documents\\Fouilles_donnee_R\\3_mini-projets\\projet_1')
data <- read.csv("Emotion_classify_Data.csv", stringsAsFactors = FALSE)



# Prétraitement des données
clean_text <- function(text) {
  text <- tolower(text)
  text <- removePunctuation(text)
  text <- removeNumbers(text)
  text <- removeWords(text, stopwords("en"))
  text <- stripWhitespace(text)
  return(text)
}

data$Cleaned_Comment <- sapply(data$Comment, clean_text)

# Création d'une matrice de termes document (Document-Term Matrix) avec TF-IDF
dtm <- DocumentTermMatrix(VCorpus(VectorSource(data$Cleaned_Comment)))
tfidf <- weightTfIdf(dtm)

# Conversion en dataframe pour la modélisation
data_tfidf <- as.data.frame(as.matrix(tfidf))
data_tfidf$Emotion <- data$Emotion

# Division des données en ensembles d'apprentissage et de test
set.seed(42)
trainingIndex <- createDataPartition(data_tfidf$Emotion, p = 0.8, list = FALSE)
trainingData <- data_tfidf[trainingIndex, ]
testData <- data_tfidf[-trainingIndex, ]

# Entraînement du modèle bayésien naïf
model <- naiveBayes(Emotion ~ ., data = trainingData)

# Prédiction et évaluation du modèle
predictions <- predict(model, testData)
confMatrix <- confusionMatrix(predictions, testData$Emotion)
print(confMatrix)
