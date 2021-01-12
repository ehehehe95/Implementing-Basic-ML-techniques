#load packages
library('ggplot2') 
library('ggthemes')
library('scales')
library('dplyr')
library('mice')

train = read.csv("Y:/CAU/기초과학/titanic/train.csv",header=TRUE)
test = read.csv("Y:/CAU/기초과학/titanic/test.csv",header=TRUE)


#explore the data and deal with NA values, and ignore unneccesarry features like name, ticketnumber

#bind data to preprocess togehter

full <- bind_rows(train, test)
str(full)
summary(full)
sum(is.na(full)) #there are 682 missing values that we have to deal with
colnames(full)
dim(full) #there are 1309 datas with 13 properties

# I do not think the property "name" is important because though we can know the title of the person
# using this property, we have age, sex, sibsp and parch property which implicitly contains the same information. 
# So I would consider this property again if my model accuracy is low, and need more features.

sum(is.na(full$Name)) #0 there's no missing value in name

# to conclude information about family members, add sibsp(siblings and spouse), parch(parents, childeren) 
# and save it to FamilySize

full$Fsize <- full$SibSp + full$Parch + 1
colnames(full)

# Use ggplot2 to visualize the relationship between family size & survival
ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') +
  theme_few()
# I would not make this Fsize into another group since I am going to use random forest

#Working with cabin

full$Cabin
sum(is.na(full$Cabin)) #there's no NA value in cabin, but just empty strings("")
sum(full$Cabin=="") #there are 1014 data without a cabin number

# I want to ignore this feature since we have a information of class of the passengersa and there's too many missing data
# But, even we have small number of data the feature which area the cabin was located would play important rule in survival
# So for now, I will categorize this data into alphabet group with the starting alphabet
# The property which seat they were seated in the area would matter too, but the data is sparse, and we are using random forest
# I can not make the specific criteria that divides the group. I will try to visualize and find the tendency, but if it
# does not work well, will just use only the alphabet

# use sub for people who own more than one cabin
full$Cabin <- sub(" .*", "", full$Cabin)
full$Deck <- substr(full$Cabin,1,1)
full$Seat <- as.numeric(substr(full$Cabin,2,4 ))

#investigating whether seat number matters using visualization
DeckC <- full$Deck=="C"
plot(full$Survived[DeckC],full$Seat[DeckC])
#there's no relation with seat number so we will just use Deck

# Show number of missing Age values
sum(is.na(full$Age)) #there are 263 missing values

#I thought name is not important at all, but in this case I have no way to predict the age.
#If I do not use name, than I only can fill NA values by putting in average age values
#So we will go back to name part, and follow the notebook of Megan L. Risdal

#his idea is to predict age value by other features using mice, so I will follow his code for next lines

# Grab title from passenger names
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

# Show title counts by sex
table(full$Sex, full$Title)

# Titles with very low cell counts to be combined to "rare" level
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Also reassign mlle, ms, and mme accordingly
full$Title[full$Title == 'Mlle']        <- 'Miss' 
full$Title[full$Title == 'Ms']          <- 'Miss'
full$Title[full$Title == 'Mme']         <- 'Mrs' 
full$Title[full$Title %in% rare_title]  <- 'Rare Title'

# Show title counts by sex again
table(full$Sex, full$Title)
full$Surname <- sapply(full$Name,  
                       function(x) strsplit(x, split = '[,.]')[[1]][1])

#predict age using mice

# Make variables factors into factors
factor_vars <- c('PassengerId','Pclass','Sex',
                 'Title','Surname','Family','FsizeD')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

# Set a random seed
set.seed(129)

# Perform mice imputation, excluding certain less-than-useful variables:
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf') 

# Save the complete output 
mice_output <- complete(mice_mod)

#fill NA value with mice output
full$Age <- mice_output$Age

# Show number of missing Age values
sum(is.na(full$Age)) #no more missing value

# split the data to train and test again, use random forest
train <- full[1:891,]
test <- full[892:1309,]
library('randomForest') # classification algorithm

colnames(train)
#do not use unnecessary features like name, ticket number, seat, Embarked etc
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                           Fare + Deck + Title + Fsize,
                         data = train)
#plot errors of overall(black), survived(green), died(red)
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

# plot importance(used code of Megan L. Risdal)
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()

#predict test data from our model

prediction <- predict(rf_model, test)
colnames(prediction) #NULL
class(prediction)
prediction #found NA data in 1044
test[153,] #there's NA value in Fare, maybe this is the reason why it returned NA
sum(is.na(test$Fare)) #there is only one NA value, since the sum is 1 so I will fill this value with mean of fare
test$Fare[153]<-mean(full$Fare, na.rm = TRUE)
sum(is.na(test$Fare)) #no more NA value

#try predicting again
prediction <- predict(rf_model, test)
output <- data.frame(PassengerID = test$PassengerId, Survived = prediction) 


# export to csv to test accuracy on kaggle 
write.csv(output, file = '20140588Young.csv', row.names = F)

# I got score of 0.75358 on the Kaggle