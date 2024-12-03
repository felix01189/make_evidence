import json
import argparse
import openai
import time
from tqdm import tqdm
from collections import Counter
import sqlite3
import glob
import os
import csv
import io

openai.api_key = ""

def parse_option():
    parser = argparse.ArgumentParser("command line arguments")
    parser.add_argument("--dataset_json_path", type=str)
    parser.add_argument("--db_path", type=str)
    parser.add_argument("--output_path", type=str)

    opt = parser.parse_args()

    return opt

def make_prompt(question, schema, schema_description):
    prompt = """Your role is an evidence generator to assist with text-to-SQL operations. You must create the external knowledge for text-to-SQL operations as evidence.
    
Purpose: Create an evidence to aid text-to-SQL tasks
action
  1. Please refer to the given question, evidence, and SQL pair and the DB Schema of samples, and schema description.
  2. For the given question, schema and schema description, generate evidence in one sentence to help text-to-sql.
  3. The length of the evidence should be as short as possible and does not need to be generated if unnecessary.
  4. Skip the description and just print out evidence.

### samples ####################################################
1. Question, evidence, and SQL pair samples
{
    {
        "question": "How many apps were last updated in January of 2018? Please write one translated review with positive sentiment for each app, if there's any.",
        "evidence": "updated in January of 2018 refers to Last Updated BETWEEN 'January 1, 2018' and 'January 31, 2018';",
        "SQL": "SELECT DISTINCT Translated_Review FROM user_reviews WHERE App IN ( SELECT App FROM playstore WHERE `Last Updated` BETWEEN 'January 1, 2018' AND 'January 31, 2018' ) AND Sentiment = 'Positive'"
    },
    {
        "question": "How many users mildly likes the 7 Minute Workout app and when was it last updated?",
        "evidence": "mildly likes the app refers to Sentiment_Polarity> = 0 and Sentiment_Polarity<0.5;",
        "SQL": "SELECT COUNT(T2.Sentiment_Polarity), T1.\"Last Updated\" FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = '7 Minute Workout' AND T2.Sentiment_Polarity BETWEEN 0 AND 0.5"
    },
    {
        "question": "How many users holds neutral attitude towards the HTC Weather app? Indicate the app's rating on the Google Play Store.",
        "evidence": "user holds neutral attitude refers to Sentiment = 'Neutral';",
        "SQL": "SELECT COUNT(T1.Rating), T1.Rating FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = 'HTC Weather' AND T2.Sentiment = 'Neutral'"
    },
    {
        "question": "What is the name and category of the app with the highest amount of -1 sentiment polarity score?",
        "evidence": "highest amount of -1 sentiment polarity score refers to MAX(Count(Sentiment_Polarity = 1.0))",
        "SQL": "SELECT DISTINCT T1.App, T1.Category FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T2.Sentiment_Polarity = '-1.0'"
    },
    {
        "question": "What is the average sentiment polarity score of the Cooking Fever app? Indicate the age group that the app is targeted at.",
        "evidence": "average sentiment polarity score = AVG(Sentiment_Polarity); age group the app is target at refers to Content Rating;",
        "SQL": "SELECT AVG(T2.Sentiment_Polarity), T1.\"Content Rating\" FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = 'Cooking Fever'"
    },
    {
        "question": "What is the lowest sentiment polarity score of the Basketball Stars app for people who dislikes the app pretty much and how many downloads does it have?",
        "evidence": "lowest sentiment polarity score refers to MIN(Sentiment_Polarity); user dislike the app pretty much refers to Sentiment_Polarity<-0.5; number of downloads it has refers to installs;",
        "SQL": "SELECT MIN(T2.Sentiment_Polarity), T1.Installs FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = 'Basketball Stars'"
    },
    {
        "question": "For the Akinator app, how many reviews have sentiment subjectivity of no more than 0.5 and what is its current version?",
        "evidence": "Sentiment_Subjectivity<0.5; current version refers to Current Ver;",
        "SQL": "SELECT COUNT(T2.Sentiment_Subjectivity), T1.\"Current Ver\" FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = 'Akinator' AND T2.Sentiment_Subjectivity < 0.5"
    },
    {
        "question": "How many apps have rating of 5?",
        "evidence": "FALSE;",
        "SQL": "SELECT COUNT(App) FROM playstore WHERE Rating = 5"
    },
    {
        "question": "What are the top 5 installed free apps?",
        "evidence": "free app refers to price = 0; most installed app refers to MAX(Installs);",
        "SQL": "SELECT App FROM playstore WHERE Price = 0 ORDER BY CAST(REPLACE(REPLACE(Installs, ',', ''), '+', '') AS INTEGER) DESC LIMIT 5"
    },
    {
        "question": "Name the top 10 most reviewed apps.",
        "evidence": "most reviewed app refers to MAX(Reviews);",
        "SQL": "SELECT DISTINCT App FROM playstore ORDER BY Reviews DESC LIMIT 10"
    },
    {
        "question": "How many of the users hold neutral attitude on \"10 Best Foods for You\" app and what category is this app?",
        "evidence": "neutral attitude refers to Sentiment = 'Neutral';",
        "SQL": "SELECT COUNT(T2.App), T1.Category FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = '10 Best Foods for You' AND T2.Sentiment = 'Neutral'"
    },
    {
        "question": "What are the apps that users pretty like this app and how many installs amount of these apps?",
        "evidence": "users pretty much likes the app refers to Sentiment_Polarity = 'Positive';",
        "SQL": "SELECT DISTINCT T1.App, T1.Installs FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T2.Sentiment_Polarity > 0"
    },
    {
        "question": "List apps whose rating is 3.9 and state the translated review of each app.",
        "evidence": "lowest rating refers to Rating = 1;",
        "SQL": "SELECT T1.App, T2.Translated_Review FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Rating = 3.9"
    },
    {
        "question": "How many apps that are only compatible with Android ver 8.0 and above? List down the users' sentiment of these apps.",
        "evidence": "compatible with android refers to Android Ver; Android Ver\" = '8.0 and up';",
        "SQL": "SELECT DISTINCT Sentiment FROM user_reviews WHERE App IN ( SELECT App FROM playstore WHERE `Android Ver` = '8.0 and up' )"
    },
    {
        "question": "Which apps have multiple genres and what is the total sentiment subjectivity of these apps?",
        "evidence": "multiple genres refers to COUNT(Genres>1; total sentiment subjectivity = Sum(Sentiment_Subjectivity);",
        "SQL": "SELECT SUM(T2.Sentiment_Subjectivity) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Genres > 1"
    },
    {
        "question": "Which apps have not been updated since year 2015 and what kind of sentiment users hold on it?",
        "evidence": "since year 2015 refers to \"Last Updated\"<'January 1, 2015';",
        "SQL": "SELECT DISTINCT App, Sentiment FROM user_reviews WHERE App IN ( SELECT App FROM playstore WHERE CAST(SUBSTR('Last Updated', -4, 4) AS INTEGER) < 2015 )"
    },
    {
        "question": "What is the total installs of apps with content rating of adults only 18+ and what are the translated reviews of it?",
        "evidence": "total installs = SUM(Installs);",
        "SQL": "SELECT SUM(T1.Installs), T2.Translated_Review FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.\"Content Rating\" = 'Adults only 18+'"
    },
    {
        "question": "Which of the app is the best selling app and what is the sentiments polarity of it?",
        "evidence": "best selling app = MAX(MULTIPLY(Price, Installs));",
        "SQL": "SELECT T1.App, T2.Sentiment_Polarity FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App ORDER BY T1.Price * CAST(REPLACE(REPLACE(Installs, ',', ''), '+', '') AS INTEGER) DESC LIMIT 1"
    },
    {
        "question": "What is the average rating of comic category apps? How many users hold positive attitude towards this app?",
        "evidence": "average rating = AVG(Rating where Category = 'COMICS'); number of users who hold a positive attitude towards the app refers to SUM(Sentiment = 'Positive');",
        "SQL": "SELECT AVG(T1.Rating) , COUNT(CASE WHEN T2.Sentiment = 'Positive' THEN 1 ELSE NULL END) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Category = 'COMICS'"
    },
    {
        "question": "What is the rating for \"Draw A Stickman\"?",
        "evidence": "Draw A Stickman refers to App = 'Draw A Stickman';",
        "SQL": "SELECT Rating FROM playstore WHERE APP = 'Draw A Stickman'"
    },
    {
        "question": "How many of the reviews for the app \"Brit + Co\" have a comment?",
        "evidence": "Brit + Co refers to App = 'Brit + Co'; comment refers to Translated Review NOT null;",
        "SQL": "SELECT COUNT(App) FROM user_reviews WHERE App = 'Brit + Co' AND Translated_Review IS NOT NULL"
    },
    {
        "question": "List the top 5 shopping apps with the most reviews.",
        "evidence": "shopping apps refers to Genre = 'Shopping'; most reviews refers to MAX(Reviews);",
        "SQL": "SELECT DISTINCT App FROM playstore WHERE Genres = 'Shopping' GROUP BY App ORDER BY COUNT(App) DESC LIMIT 5"
    },
    {
        "question": "How many neutral reviews does the app \"Dino War: Rise of Beasts\" have?",
        "evidence": "neutral reviews refers to Sentiment = 'Neutral';",
        "SQL": "SELECT COUNT(App) FROM user_reviews WHERE App = 'Dino War: Rise of Beasts' AND Sentiment = 'Neutral'"
    },
    {
        "question": "What are the apps with only 5,000+ installs?",
        "evidence": "Installs = '5,000+';",
        "SQL": "SELECT DISTINCT App FROM playstore WHERE Installs = '5,000+'"
    },
    {
        "question": "List all the negative comments on the \"Dog Run - Pet Dog Simulator\" app.",
        "evidence": "negative comment refers to Sentiment = 'Negative';",
        "SQL": "SELECT Translated_Review FROM user_reviews WHERE App = 'Dog Run - Pet Dog Simulator' AND Sentiment = 'Negative'"
    },
    {
        "question": "Which free app has the most Negative comments?",
        "evidence": "paid app refers to Type = 'Paid'; negative comment refers to Sentiment = 'Negative'; paid app with most negative comments refers to MAX(COUNT(Sentiment = 'Negative')) where Type = 'Paid';",
        "SQL": "SELECT T1.App FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Type = 'Free' AND T2.Sentiment = 'Negative' GROUP BY T1.App ORDER BY COUNT(T2.Sentiment) DESC LIMIT 1"
    },
    {
        "question": "How many negative comments are there in all the apps with 100,000,000+ installs?",
        "evidence": "negative comment refers to Sentiment = 'Negative'; Installs = '100,000,000+';",
        "SQL": "SELECT COUNT(T2.Sentiment) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Installs = '100,000,000+' AND T2.Sentiment = 'Negative'"
    },
    {
        "question": "What are the content ratings for the apps that have \"gr8\" in their comments?",
        "evidence": "app with gr8 in their comments refers to Translated_Review LIKE '%gr8%';",
        "SQL": "SELECT DISTINCT T1.`Content Rating` FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T2.Translated_Review LIKE '%gr8%'"
    },
    {
        "question": "What is the total Sentiment polarity score of the most expensive app?",
        "evidence": "total sentiment polarity score = sum(Sentiment_Polarity); most expensive app refers to MAX(Price);",
        "SQL": "SELECT SUM(T2.Sentiment_Polarity) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Price = ( SELECT MAX(Price) FROM playstore )"
    },
    {
        "question": "What is the rating for \"Garden Coloring Book\"? List all of its reviews.",
        "evidence": "Golfshot Plus: Golf GPS refers to App = 'Golfshot Plus: Golf GPS'; review refers to Translated_Review;",
        "SQL": "SELECT T1.Rating, T2.Translated_Review FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = 'Garden Coloring Book'"
    },
    {
        "question": "Which Photography app has the highest total Sentiment subjectivity score?",
        "evidence": "Photography app refers to Genre = 'Photography'; highest total sentiment subjectivity score = MAX(sum(Sentiment_Subjectivity));",
        "SQL": "SELECT T1.App FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Genres = 'Photography' GROUP BY T1.App ORDER BY SUM(T2.Sentiment_Subjectivity) DESC LIMIT 1"
    },
    {
        "question": "List all the comments on the lowest rated Mature 17+ app.",
        "evidence": "comments refers to Translated_Review; lowest rated refers to Rating = 1; Mature 17+ refers to Content Rating = 'Mature 17+ ';",
        "SQL": "SELECT T2.Translated_Review FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.\"Content Rating\" = 'Mature 17+' ORDER BY T1.Rating LIMIT 1"
    },
    {
        "question": "What is the number of installments of the app with the highest total Sentiment polarity score?",
        "evidence": "installments refers to Installs; highest total sentiment polarity score = MAX(SUM(Sentiment_Polarity));",
        "SQL": "SELECT T1.Installs FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App GROUP BY T1.App ORDER BY SUM(T2.Sentiment_Polarity) DESC LIMIT 1"
    },
    {
        "question": "What is the number of neutral comments from all the weather apps?",
        "evidence": "neutral comments refers to Sentiment = 'Neutral'; weather app refers to Genre = 'Weather';",
        "SQL": "SELECT COUNT(T2.Sentiment) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Genres = 'Weather' AND T2.Sentiment = 'Neutral'"
    },
    {
        "question": "Which 1,000,000,000+ intalls apps has the most no comment reviews?",
        "evidence": "no comment refers to Translated_Review = 'nan'; most no comment reviews = (MAX(COUNT(Translated_Review = 'nan')));",
        "SQL": "SELECT T1.App FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Installs = '1,000,000+' AND T2.Translated_Review = 'nan' GROUP BY T1.App ORDER BY COUNT(T2.Translated_Review) DESC LIMIT 1"
    },
    {
        "question": "What is the rating and the total Sentiment subjectivity score of \"Onefootball - Soccer Scores\"?",
        "evidence": "Onefootball - Soccer Scores refers to App = 'Onefootball - Soccer Scores';",
        "SQL": "SELECT T1.Rating, SUM(T2.Sentiment_Subjectivity) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = 'Onefootball - Soccer Scores'"
    },
    {
        "question": "What percentage of no comment reviews are from \"Teen\" content rating apps?",
        "evidence": "no comment refers to Translated_Review = 'nan'; percentage = DIVIDE((SUM(Content Rating = 'Teen')), COUNT(*));",
        "SQL": "SELECT CAST(COUNT(CASE WHEN T1.`Content Rating` = 'Teen' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T1.App) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T2.Translated_Review = 'nan'"
    },
    {
        "question": "Which apps have 5 rating? List out then application name.",
        "evidence": "application name refers to App;",
        "SQL": "SELECT DISTINCT App FROM playstore WHERE Rating = 5"
    },
    {
        "question": "Which apps have been reviewed more than 75 000 000 times and the content is suitable for teenagers?",
        "evidence": "Reviews>75000000; suitable for teenagers refers to Content Rating = 'Teen';",
        "SQL": "SELECT DISTINCT App FROM playstore WHERE Reviews > 75000000 AND `Content Rating` = 'Teen'"
    },
    {
        "question": "List out genre that have downloads more than 1000000000.",
        "evidence": "downloads and installs are synonyms; Installs = '1,000,000,000+';",
        "SQL": "SELECT Genres FROM playstore WHERE Installs = '1,000,000,000+' GROUP BY Genres"
    },
    {
        "question": "What is the average price for a dating application?",
        "evidence": "average price = AVG(Price where Genre = 'Dating'); dating application refers to Genre = 'Dating';",
        "SQL": "SELECT AVG(Price) FROM playstore WHERE Genres = 'Dating'"
    },
    {
        "question": "What is the average download for entertainment apps with size no more than 1.0 M?",
        "evidence": "downloads and installs are synonyms; entertainment apps refers to Category = 'ENTERTAINMENT';",
        "SQL": "SELECT AVG(CAST(REPLACE(REPLACE(Installs, ',', ''), '+', '') AS INTEGER)) FROM playstore WHERE Category = 'ENTERTAINMENT' AND Size < '1.0M'"
    },
    {
        "question": "What is the average review number for application with 5 rating?",
        "evidence": "average review = AVG(Review); application refers to app; Rating = 5;",
        "SQL": "SELECT AVG(Reviews) FROM playstore WHERE Rating = 5"
    },
    {
        "question": "List out the top 3 genre for application with a sentiment review greater than 0.5.",
        "evidence": "sentiment review refers to Sentiment_Polarity; Sentiment_Polarity>0.5;",
        "SQL": "SELECT Genres FROM playstore WHERE App IN ( SELECT App FROM user_reviews WHERE Sentiment = 'Positive' AND Sentiment_Polarity > 0.5 ORDER BY Sentiment_Polarity DESC LIMIT 3 )"
    },
    {
        "question": "What is the percentage of application with 4.7 rating having more positives sentiment than negative sentiment?",
        "evidence": "percentage = DIVIDE(SUBTRACT(SUM(Sentiment = 'Positive')), (SUM(Sentiment = 'Negative')), SUM(Sentiment = 'Negative')) as percentage; having more positive sentiment than negative sentiment refers to Sentiment = 'Positive'>Sentiment = 'Negative';",
        "SQL": "SELECT CAST(COUNT(CASE WHEN ( SELECT COUNT(CASE WHEN Sentiment = 'Positive' THEN 1 ELSE NULL END) - COUNT(CASE WHEN Sentiment = 'Negative' THEN 1 ELSE NULL END) FROM user_reviews GROUP BY App ) > 0 THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T2.Sentiment) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Rating = 4.7"
    },
    {
        "question": "List down app that does not have negative sentiment and give their average rating?",
        "evidence": "doest not have negative sentiment refers to Sentiment! = 'Negative'; average = AVG(Sentiment_Polarity);",
        "SQL": "SELECT T1.App, AVG(T2.Sentiment_Polarity) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T2.Sentiment != 'Negative' GROUP BY T1.App"
    },
    {
        "question": "List down application that have not been updated since 2015. What is the percentage of this application having more negative sentiment than positive sentiment?",
        "evidence": "percentage = DIVIDE(SUBTRACT(SUM(Sentiment = 'Positive')), (SUM(Sentiment = 'Negative'))), (SUM(Sentiment = 'Negative')) as percent; Last Updated>'2015';",
        "SQL": "SELECT CAST((( SELECT COUNT(*) Po FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE SUBSTR(T1.\"Last Updated\", -4, 4) > '2015' AND T2.Sentiment = 'Positive' ) - ( SELECT COUNT(*) Ne FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE SUBSTR(T1.\"Last Updated\", -4, 4) > '2015' AND T2.Sentiment = 'Negative' )) AS REAL) * 100 / ( SELECT COUNT(*) NUM FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE SUBSTR(T1.\"Last Updated\", -4, 4) > '2015' )"
    },
    {
        "question": "What is the percentage for free application with a rating 4.5 and above have not been updated since 2018?",
        "evidence": "paid refers to Type = 'Paid'; application refers to App; Rating>4.5; Last Updated>'2018; percentage = DIVIDE(SUM(Genres = 'Mature 17+' and Rating>4.5 and\u00a0substr(\"Last Updated\",-4,4)>'2018' )), (COUNT(App)) as percent;",
        "SQL": "SELECT CAST(SUM(CASE WHEN SUBSTR('Last Updated', -4) > '2018' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(App) PER FROM playstore WHERE Type = 'Free' AND Rating >= 4.5"
    },
    {
        "question": "What genre does Honkai Impact 3rd belong to?",
        "evidence": "Honkai Impact 3rd is the App;",
        "SQL": "SELECT DISTINCT Genres FROM playstore WHERE App = 'Honkai Impact 3rd'"
    },
    {
        "question": "List down the rating for the App Learn C++.",
        "evidence": "FALSE;",
        "SQL": "SELECT DISTINCT Rating FROM playstore WHERE App = 'Learn C++'"
    },
    {
        "question": "What is the average price of games belonging in the arcade genre which has a content rating of Everyone 10+?",
        "evidence": "average price = AVG(Price);",
        "SQL": "SELECT AVG(Price) FROM playstore WHERE 'Content Rating' = 'Everyone 10+' AND Genres = 'Arcade'"
    },
    {
        "question": "How much is the size of Browser 4G and how many users have a pretty positive favorability on it?",
        "evidence": "Browser 4G is the App; pretty positive favorability refers to Sentiment_Polarity score = 0.5",
        "SQL": "SELECT T1.Size, COUNT(T1.App) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = 'Browser 4G' AND T2.Sentiment_Polarity >= 0.5"
    },
    {
        "question": "Name the Apps with a sentiment objectivity of 0.3 and include their number of installs.",
        "evidence": "FALSE;",
        "SQL": "SELECT DISTINCT T1.App, T1.Installs FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T2.Sentiment_Polarity = 0.3"
    },
    {
        "question": "How much is the average sentiment polarity score of Golf GPS Rangefinder: Golf Pad and what is it's rating in the Google Play Store?",
        "evidence": "average sentiment polarity score = AVG(Sentiment_Polarity); Golf GPS Rangefinder: Golf Pad\u00a0 is the App;",
        "SQL": "SELECT AVG(T2.Sentiment_Polarity), T1.Rating FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = 'Golf GPS Rangefinder: Golf Pad'"
    },
    {
        "question": "List the top 5 lowest rated puzzle games and count the number of negative sentiments the games received.",
        "evidence": "lowest rating refers to MIN(Rating); puzzle is the genre;",
        "SQL": "SELECT T1.App, COUNT(T1.App) COUNTNUMBER FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T2.Sentiment = 'Negative' GROUP BY T1.App ORDER BY T1.Rating LIMIT 5"
    },
    {
        "question": "What is the percentage ratio between positive sentiments and negative sentiments that are in Fate/Grand Order? Also indicate the current version.",
        "evidence": "Fate/Grand Order is the App; percentage ratio = MULTIPLY(DIVIDE((SUM(Sentiment = 'Positive')), (SUM(Sentiment = 'Negative'))), 100);",
        "SQL": "SELECT CAST(SUM(CASE WHEN T2.Sentiment = 'Positive' THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(CASE WHEN T2.Sentiment = 'Negative' THEN 1 ELSE 0 END), T1.`Current Ver` FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = 'Fate/Grand Order (English)' AND T1.`Current Ver` = '1.18.0'"
    },
    {
        "question": "Indicate the number of installs and include the percentage of positive sentiments of FREEDOME VPN Unlimited anonymous Wifi Security.",
        "evidence": "FREEDOME VPN Unlimited anonymous Wifi Security is the App; percentage = MULTIPLY(DIVIDE((SUM(Sentiment = 'Positive')), (COUNT(*))), 100)",
        "SQL": "SELECT T1.Installs , CAST(SUM(CASE WHEN T2.Sentiment = 'Positive' THEN 1 ELSE 0 END) * 100 / SUM(CASE WHEN T2.Sentiment IS NOT NULL THEN 1.0 ELSE 0 END) AS REAL) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = 'FREEDOME VPN Unlimited anonymous Wifi Security'"
    },
    {
        "question": "For the Honkai Impact 3rd App, what is the highest sentiment polarity score and what genre does it belong to?",
        "evidence": "highest sentiment polarity score refers to MAX(Sentiment_Polarity);",
        "SQL": "SELECT MAX(T2.Sentiment_Polarity), T1.Genres FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = 'Honkai Impact 3rd' AND T2.Sentiment_Polarity > 0.5 GROUP BY T1.Genres"
    },
    {
        "question": "What is the rating of Dragon Ball Legends and how many users dislike this App?",
        "evidence": "Dragon Ball Legends is the app; users who dislikes the app refers to Sentiment_Polarity<-0.5;",
        "SQL": "SELECT T1.Rating, COUNT(T2.Sentiment_Polarity) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.App = 'Dragon Ball Legends' AND CAST(Sentiment_Polarity AS INTEGER) < -0.5"
    },
    {
        "question": "Which education App has the worst rating and state the translated review if available.",
        "evidence": "education App refers to Category = 'EDUCATION'; worst rated app refers to Rating = 1;",
        "SQL": "SELECT T1.App, T2.Translated_Review FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Category = 'EDUCATION' GROUP BY T1.App, T2.Translated_Review ORDER BY T1.Rating ASC LIMIT 1"
    },
    {
        "question": "List all free sports Apps and their translated review.",
        "evidence": "paid sports Apps refers to type = 'Paid' and Category = 'SPORTS';",
        "SQL": "SELECT T1.App, T2.Translated_Review FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Type = 'Free' AND T1.Category = 'SPORTS'"
    },
    {
        "question": "Among the role playing game genre, how many are targeted to teens and what is their average sentiment polarity score?",
        "evidence": "targeted to teen refers to Content Rating = 'Teen'; average = AVG(Sentiment_Polarity);",
        "SQL": "SELECT COUNT(T1.App), AVG(T2.Sentiment_Polarity) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.\"Content Rating\" = 'Teen' AND T1.Genres = 'Role Playing'"
    },
    {
        "question": "What is the average rating of Apps falling under the racing genre and what is the percentage ratio of positive sentiment reviews?",
        "evidence": "average rating = AVG(Rating); percentage = MULTIPLY(DIVIDE((SUM(Sentiment = 'Positive')), (COUNT(*)), 100));",
        "SQL": "SELECT AVG(T1.Rating), CAST(COUNT(CASE WHEN T2.Sentiment = 'Positive' THEN 1 ELSE NULL END) AS REAL) * 100 / COUNT(T2.Sentiment) FROM playstore AS T1 INNER JOIN user_reviews AS T2 ON T1.App = T2.App WHERE T1.Genres = 'Racing'"
    }
}

2. DB Schema of Samples
{
    CREATE TABLE playstore (
        App              TEXT,
        Category         TEXT,
        Rating           REAL,
        Reviews          INTEGER,
        Size             TEXT,
        Installs         TEXT,
        Type             TEXT,
        Price            TEXT,
        [Content Rating] TEXT,
        Genres           TEXT
    );

    CREATE TABLE user_reviews (
        App                    TEXT REFERENCES playstore (App),
        Translated_Review      TEXT,
        Sentiment              TEXT,
        Sentiment_Polarity     TEXT,
        Sentiment_Subjectivity TEXT
    );
}

3. Schema Description
{
    playstore
    {
    original_column_name,column_name,column_description,data_format,value_description
    App,,Application name,text,
    Category,,Category the app belongs to,text,"FAMILY 18%
    GAME 11%
    Other (7725) 71%"
    Rating,,Overall user rating of the app (as when scraped),real,
    Reviews,,Number of user reviews for the app (as when scraped),integer,
    Size,,Size of the app (as when scraped),text,"Varies with device 16%
    11M 2%
    Other (8948) 83%"
    Installs,,Number of user downloads/installs for the app (as when scraped),text,"1,000,000+ 15%
    10,000,000+ 12%
    Other (8010) 74%"
    Type,,Paid or Free,text,"Only has 'Paid' and 'Free'
    Free 93%
    Paid 7%"
    Price,,Price of the app (as when scraped),text,"0 93%
    $0.99 1%
    Other (653) 6%

    commonsense evidence:
    Free means the price is 0."
    Content Rating,,Age group the app is targeted at - Children / Mature 21+ / Adult,text,"Everyone 80%
    Teen 11%
    Other (919) 8%

    commonsense evidence:
    According to Wikipedia, the different age groups are defined as:
    Children: age<12~13
    Teen: 13~19
    Adult/Mature: 21+
    "
    Genres,,An app can belong to multiple genres (apart from its main category).,text,"Tools 8%
    Entertainment 6%
    Other (9376) 86%"
    }
    user_reviews
    {
    original_column_name,column_name,column_description,data_format,value_description
    App,,Name of app,text,
    Translated_Review,,User review (Preprocessed and translated to English),text,"nan 42%
    Good 0%
    Other (37185) 58%"
    Sentiment,,Overall user rating of the app (as when scraped),text,"commonsense evidence:
    Positive: user holds positive attitude towards this app
    Negative: user holds positive attitude / critizes on this app
    Neural: user holds neural attitude
    nan: user doesn't comment on this app."
    Sentiment_Polarity,Sentiment Polarity,Sentiment polarity score,text,"commonsense evidence:
    • score >= 0.5 it means pretty positive or pretty like this app.
    • 0 <= score < 0.5: it means user mildly likes this app.
    • -0.5 <= score < 0: it means this user mildly dislikes this app or slightly negative attitude
    • score <-0.5: it means this user dislikes this app pretty much."
    Sentiment_Subjectivity,Sentiment Subjectivity,Sentiment subjectivity score,text,"commonsense evidence:
    more subjectivity refers to less objectivity, vice versa."
    }
}
##################################################################

### question ####################################################
"""
    prompt += f"""1. question
{{
    "question": "{question}",
    "evidence": 
}}

2. schema of question
{{
    {schema}
}}

3. Schema Description
{{
    {schema_description}
}}

"""
    return prompt


def generate_reply(input):
    completions = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        model="gpt-4o-mini",
        # model="gpt-4o",
        messages=input,
        # top_p=0.5
        temperature=0.7
        # stop=["Q:"]
    )
    
    return completions.choices[0].message.content


def generate_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    schemas = cursor.fetchall()
    schema = ""
    for sc in schemas:
        schema += sc[0]
    return schema

def read_schema_description(db_path):
    files = os.listdir(db_path)

    schema_description = ""
    for csv_file in files:
        schema_description += (csv_file + '\n{\n')
        csv_output = io.StringIO()
        
        with open(db_path + '/' + csv_file, 'rb') as f:
            content = f.read().replace(b'\x00', b' ')
            content = content.decode('cp1252') 

        csv_reader = csv.reader(io.StringIO(content))
        writer = csv.writer(csv_output)
        
        for row in csv_reader:
            writer.writerow(row)

        schema_description += (csv_output.getvalue() + '}\n')
    return schema_description


if __name__ == "__main__":
    opt = parse_option()
    print(opt)
    with open(opt.dataset_json_path, encoding='utf-8') as f:
        question_json_all = json.load(f)
    res = []

    for i, data in enumerate(tqdm(question_json_all)):
        schema = generate_schema(f"{opt.db_path}/{data['db_id']}/{data['db_id']}.sqlite")
        schema_description = read_schema_description(f"{opt.db_path}/{data['db_id']}/database_description")
        prompt = make_prompt(data["question"], schema, schema_description)
        evidence = None
        while evidence is None:
            try:
                evidence = generate_reply([{"role": "user", "content": prompt}])
            except:
                print('api error, wait for 3 seconds and retry...')
                time.sleep(3)
                pass
               
        evidence = evidence.replace("evidence: ","")
        data["evidence"] = evidence
        data["text"] = evidence + " " + data["text"]
        res.append(data)
        
    with open(opt.output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
