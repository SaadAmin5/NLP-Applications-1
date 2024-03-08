#importing packages
import pandas as pd
import spacy
nlp=spacy.load('en_core_web_sm')

#importing dataset using pandas and naming it data
data=pd.read_csv("amazon_product_reviews.csv", low_memory=False)
print(data.head()) #printing first 5 rows of data

print('Shape of data is: ', data.shape) #finding number of rows and columns in data

print('\n')

print(data.columns) #getting column names

print('\n')

print(data.describe())  #getting general description of numeric columns

print('\n')

print(data.isna().sum())  #finding total number of missing values in each column

print('\n')

# 2.1 selecting the 'review.text' column
reviews_data=data['reviews.text']
print(reviews_data)

clean_data= data.copy()  #making a copy of original data

#2.2 removing missing values from review.text column
clean_data=clean_data.dropna(subset=['reviews.text'])
print(clean_data.head())

print('\n')

print(clean_data.isna().sum()) #reviews.text has 0 missing value now

print('\n')

# removing stopwords from reviews.text without missing values 
#.is_stop is used for string columns that have no missing values
def remove_stopwords(text1):
    
    list1=[]
    doc=nlp(text1)
    
    for token in doc:
        
        if not token.is_stop:
            list1.append(token.text)
            
    return ' '.join(list1)     #this is to convert list into string

clean_data['reviews.text']=clean_data['reviews.text'].apply(remove_stopwords)
print(clean_data['reviews.text'].head())  #you can see stopwords removed from reviews.text column
print('\n')

# doing some more data processing
clean_data.drop_duplicates(inplace=True)   #dropping duplicate rows

print(clean_data.shape)  #there was 1 duplicate row in the dataset which is removed
print('\n')

print(clean_data.isna().sum())  #reviews.dateAdded, reviews.didPurchase, reviews.id, reviews.userCity, reviews.userProvince contains major missing values, so its better to drop them
print('\n')

#dropping reviews.dateAdded, reviews.didPurchase, reviews.id, reviews.userCity, reviews.userProvince columns as they contain major missing values
clean_data.drop(columns=['reviews.dateAdded','reviews.didPurchase', 'reviews.id',
                        'reviews.userCity', 'reviews.userProvince'], inplace=True, axis=1)
print(clean_data.head())
print('\n')

print(clean_data.shape)  #now we are left with 16 columns
print('\n')

#dropping all missing values from name column
clean_data.dropna(subset=['name'], inplace=True)

print(clean_data.isna().sum())  #missing values in 'name' column is 0 now
print('\n')

print(clean_data['reviews.doRecommend'].value_counts())  #the most repeated value in 'reviews.doRecommend' is True, so we will replace missing value with True
print('\n')

#replacing missing values in 'reviews.doRecommend' with most repeated value 'True'
clean_data['reviews.doRecommend']=clean_data['reviews.doRecommend'].fillna(clean_data['reviews.doRecommend'].mode()[0])

#replacing missing values in 'reviews.numHelpful' with most repeated value 0
clean_data['reviews.numHelpful']=clean_data['reviews.numHelpful'].fillna(clean_data['reviews.numHelpful'].mode()[0])

#replacing missing values in 'reviews.rating' with most repeated value 5
clean_data['reviews.rating']=clean_data['reviews.rating'].fillna(clean_data['reviews.rating'].mode()[0])

print(clean_data.isna().sum())  #there's no missing value in 'reviews.doRecommend', 'reviews.numHelpful', 'reviews.rating'
print('\n')

#replacing missing values with the previous row using ffill method
clean_data[['asins', 'reviews.title', 'reviews.username', 'reviews.date']]=clean_data[['asins', 'reviews.title', 'reviews.username', 'reviews.date']].fillna(method='ffill') 

print(clean_data.isna().sum())  # all mising values are removed

print('\n')

#removing whitespaces and converting text to lower case in 'reviews.text', 'categories', 'reviews.title', 'reviews.username' &'name' columns   
clean_data['reviews.text']=clean_data['reviews.text'].str.strip().str.lower()
clean_data['categories']=clean_data['categories'].str.strip().str.lower()
clean_data['reviews.title']=clean_data['reviews.title'].str.strip().str.lower()
clean_data['reviews.username']=clean_data['reviews.username'].str.strip().str.lower()
clean_data['name']=clean_data['name'].str.strip().str.lower()

print(clean_data.head())  # a much cleaner dataset

print('\n')

'''
3. Creating a function for sentiment analysis. Defined a function that takes
a product review as input and predicts its sentiment.'''
import spacy
from textblob import TextBlob
# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


def find_sentiment(review):
    
    review = TextBlob(review)
    polarity_score = review.sentiment.polarity
    
    if polarity_score > 0:
        sentiment = 'positive'
    
    elif polarity_score<0:
        sentiment = 'negative'

    else:
        sentiment = 'neutral'
        
    print('Sentiment:', sentiment)

print('\n')

input_review=input('Write your product review: ')
print(find_sentiment(review=input_review))

print('\n')

'''
4. Test your model on sample product reviews: Test the sentiment
analysis function on a few sample product reviews to verify its accuracy
in predicting sentiment.
'''
# getting sample size of 10 from reviews.text for sentimental analysis
sample_reviews=clean_data['reviews.text'].sample(10, random_state=1)
print(sample_reviews)

print('\n')

#testing the model on sample reviews
print(sample_reviews.apply(find_sentiment))

print('\n')

# additional work
# finding polarity score for sample reviews
def find_polarity_score(review):
    
    review = TextBlob(review)
    polarity_score = review.sentiment.polarity
    
    return polarity_score

print(sample_reviews.apply(find_polarity_score))  #it shows polarity score greater than 0 for all sample reviews which means these are positive sentiments

print('\n')

#checking similarity of 2 product reviews
my_review_of_choice1=clean_data['reviews.text'][2]   #selecting row with index 2 of reviews.text
print(my_review_of_choice1)

my_review_of_choice2=clean_data['reviews.text'][8]   #selecting row with index 8 of reviews.text
print(my_review_of_choice2)

print('\n')

nlp = spacy.load("en_core_web_md")  #loading medium size model that include word vectors
print('Similarity between both the reviews is: ', nlp(my_review_of_choice1).similarity(nlp(my_review_of_choice2))) #it shows a lot of similarity as both are the reviews by consumers
