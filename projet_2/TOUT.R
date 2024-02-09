knitr::opts_chunk$set(echo = TRUE)

rm(list = ls())
# Charger les packages nécessaires
library("data.table")
library("tm")
library("textclean")
library("MASS")
library("text2vec")
library("spacyr")



# Charger les données
training_data <- read.csv("twitter_training.csv", header = FALSE, stringsAsFactors = FALSE)
validation_data <- read.csv("twitter_validation.csv", header = FALSE, stringsAsFactors = FALSE)

# Définir les noms de colonnes
colnames(training_data) <- c('ID','Pays','Sentiment','Texte')  # Remplacez ... par les noms de vos colonnes
colnames(validation_data) <- c('ID','Pays','Sentiment','Texte')  # Remplacez ... par les noms de vos colonnes

table(training_data$Sentiment)
table(validation_data$Sentiment)

# Fonction pour nettoyer le texte des tweets
preprocess <- function(texte) {
    texte <- gsub("https\\S*", "", texte)   # Supprimer les URLs
    texte <- gsub("@\\S*", "", texte)   # Supprimer les mentions 
    texte <- gsub("amp", "", texte)   # Supprimer les &
    texte <- gsub("[\r\n]", "", texte)   # Supprimer ce qui permet les retour a la ligne,etc
    texte <- gsub("(<U\\+[0-9A-F]+>)", "", texte) #supprimer les emojis 
    texte <- removeNumbers(texte)                  # Supprimer les nombres
    texte <- removePunctuation(texte)              # Supprimer la ponctuation
    texte <- tolower(texte)                        # Convertir en minuscules
    texte <- removeWords(texte, stopwords("en"))  # Supprimer les mots vides
    texte <- stripWhitespace(texte)                # Supprimer les espaces superflus
    return(texte)
}


# Application du nettoyage
training_data$Clean_Texte <- sapply(training_data$Texte, preprocess)
validation_data$Clean_Texte <- sapply(validation_data$Texte, preprocess)

# Afficher un aperçu des données après nettoyage
print("Données après nettoyage - Training Data")
head(training_data$Clean_Texte)

print("Données après nettoyage - Validation Data")
head(validation_data$Clean_Texte)

library(textclean)
library(MASS)
library(text2vec)

# Conversion des étiquettes de sentiment
training_data$Sentiment <- as.factor(training_data$Sentiment)
validation_data$Sentiment <- as.factor(validation_data$Sentiment)

# Afficher les étiquettes de sentiment converties
print("Étiquettes de sentiment converties - Training Data")
head(training_data$Sentiment)

print("Étiquettes de sentiment converties - Validation Data")
head(validation_data$Sentiment)