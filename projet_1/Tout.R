knitr::opts_chunk$set(echo = TRUE)

rm(list = ls())

# Chargement des librairies
library("tidyverse")
library("text")
library("kableExtra")
library("e1071")
library("tm")
library("wordcloud")
library("caret")
library("text")
library("tidytext")
library("spacyr")

setwd('C:\\Users\\gdaie\\Documents\\Fouilles_donnee_R\\3_mini-projets\\projet_1')
data <- read.csv("Emotion_classify_Data.csv", stringsAsFactors = FALSE)

#spacy_install()
spacy_initialize(model = "en_core_web_sm")

stopWords2 <- c(stopwords("english"), 'feel', 'feeling', 'really', 'time', 'im', 'know', 'make', 'little')

# Fonction de prétraitement avec lemmatisation
preprocess <- function(text) {
  # Création d'un corpus
  corpus <- VCorpus(VectorSource(text))  
  # Nettoyage des données
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, removeWords, stopWords2)

  # Lemmatisation avec spacyr
  lemmatized_text <- sapply(corpus, function(x) {
    parsed <- spacy_parse(as.character(x), lemma = TRUE)
    paste(parsed$lemma, collapse = " ")
  })

  return(lemmatized_text)
}

# Appliquer la fonction de prétraitement
data$preprocessed_comment <- sapply(data$Comment, preprocess)

data$Emotion_num <- match(data$Emotion, c('joy', 'fear', 'anger')) - 1

data$Emotion_num <- match(data$Emotion, c('joy', 'fear', 'anger')) - 1
head(data, 5)
library("caret")

# Assurez-vous que df est votre dataframe et qu'il contient les colonnes 'preprocessed_comment' et 'Emotion_num'

# Création d'un index pour le partitionnement stratifié
set.seed(42)
trainIndex <- createDataPartition(data$Emotion_num, p = 0.8, list = FALSE, times = 1)

# Séparation des données en ensembles d'entraînement et de test
X_train <- data$preprocessed_comment[trainIndex]
y_train <- data$Emotion_num[trainIndex]
X_test <- data$preprocessed_comment[-trainIndex]
y_test <- data$Emotion_num[-trainIndex]
library(text2vec)

# Assurez-vous que X_train et X_test sont disponibles et contiennent les données textuelles

# Création d'un itérateur sur les documents
it_train <- itoken(X_train, progressbar = FALSE)
it_test <- itoken(X_test, progressbar = FALSE)

# Création du vocabulaire et du vectoriseur
vocab <- create_vocabulary(it_train)
vectorizer <- vocab_vectorizer(vocab)

# Transformation des données d'entraînement et de test en utilisant TF-IDF
tfidf_transformer <- TfIdf$new()
dtm_train <- create_dtm(it_train, vectorizer) %>%
  tfidf_transformer$fit_transform()
dtm_test <- create_dtm(it_test, vectorizer) %>%
  tfidf_transformer$transform()

# Assumons que dtm_train est votre Document-Term Matrix et y_train est votre vecteur de réponse

# Entraînement du modèle Naive Bayes
NB_model1 <- naiveBayes(as.matrix(dtm_train), as.factor(y_train))

# NB_model est maintenant le modèle entraîné

# Classification
predictions1 <- predict(NB_model1, newdata = as.matrix(dtm_test), type = "class")
# Conversion des prédictions et des valeurs réelles en facteurs avec les mêmes niveaux
levels <- sort(unique(c(y_train, y_test))) # Fusion et tri des niveaux uniques dans les données d'entraînement et de test
predictions_factor1 <- factor(predictions1, levels = levels)
y_test_factor <- factor(y_test, levels = levels)

# Calcul de la matrice de confusion
conf_matrix1 <- confusionMatrix(predictions_factor1, y_test_factor)

# Affichage de la matrice de confusion
print(conf_matrix1)
#De meme avec un autre modele d'un autre package
library("naivebayes")

# Entraînement du modèle Naive Bayes
NB_model <- multinomial_naive_bayes(as.matrix(dtm_train), as.factor(y_train), laplace = 1)

# NB_model est maintenant le modèle entraîné

# Classification
predictions <- predict(NB_model, newdata = as.matrix(dtm_test), type = "class")
# Conversion des prédictions et des valeurs réelles en facteurs avec les mêmes niveaux
levels <- sort(unique(c(y_train, y_test))) # Fusion et tri des niveaux uniques dans les données d'entraînement et de test
predictions_factor <- factor(predictions, levels = levels)
y_test_factor <- factor(y_test, levels = levels)

# Calcul de la matrice de confusion
conf_matrix <- confusionMatrix(predictions_factor, y_test_factor)

# Affichage de la matrice de confusion
print(conf_matrix)