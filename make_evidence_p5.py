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
        "question": "Name movie titles released in year 1945. Sort the listing by the descending order of movie popularity.",
        "evidence": "released in the year 1945 refers to movie_release_year = 1945;",
        "SQL": "SELECT movie_title FROM movies WHERE movie_release_year = 1945 ORDER BY movie_popularity DESC LIMIT 1"
    },
    {
        "question": "State the most popular movie? When was it released and who is the director for the movie?",
        "evidence": "most popular movie refers to MAX(movie_popularity); when it was released refers to movie_release_year; director for the movie refers to director_name;",
        "SQL": "SELECT movie_title, movie_release_year, director_name FROM movies ORDER BY movie_popularity DESC LIMIT 1 "
    },
    {
        "question": "What is the name of the longest movie title? When was it released?",
        "evidence": "longest movie title refers to MAX(LENGTH(movie_title)); when it was released refers to movie_release_year;",
        "SQL": "SELECT movie_title, movie_release_year FROM movies ORDER BY LENGTH(movie_popularity) DESC LIMIT 1"
    },
    {
        "question": "Name the movie with the most ratings.",
        "evidence": "movie with the most rating refers to MAX(SUM(rating_score));",
        "SQL": "SELECT movie_title FROM movies GROUP BY movie_title ORDER BY COUNT(movie_title) DESC LIMIT 1"
    },
    {
        "question": "What is the average number of Mubi users who love movies directed by Stanley Kubrick?",
        "evidence": "average = AVG(movie_popularity); number of Mubi users who loves the movie refers to movie_popularity;",
        "SQL": "SELECT AVG(movie_popularity) FROM movies WHERE director_name = 'Stanley Kubrick'"
    },
    {
        "question": "What is the average rating for movie titled 'When Will I Be Loved'?",
        "evidence": "average rating = DIVIDE((SUM(rating_score where movie_title = 'When Will I Be Loved')), COUNT(rating_score));",
        "SQL": "SELECT AVG(T2.rating_score) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_title = 'When Will I Be Loved'"
    },
    {
        "question": "What is the user avatar url for user 41579158? What is the latest movie rated by him / her?",
        "evidence": "user avatar url refers to user_avatar_image_url; latest movie rated refers to latest rating_date;",
        "SQL": "SELECT T3.user_avatar_image_url, T3.rating_date_utc FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id INNER JOIN ratings_users AS T3 ON T3.user_id = T2.user_id WHERE T3.user_id = 41579158 ORDER BY T3.rating_date_utc DESC LIMIT 1"
    },
    {
        "question": "What is the percentage of the ratings were rated by user who was a subcriber?",
        "evidence": "user is a subscriber refers to user_subscriber = 1; percentage of ratings = DIVIDE(SUM(user_subscriber = 1), SUM(rating_score)) as percent;",
        "SQL": "SELECT CAST(SUM(CASE WHEN user_subscriber = 1 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM ratings"
    },
    {
        "question": "List all movie title rated in April 2020 from user who was a trialist.",
        "evidence": "movie title rated in April 2020 refers to rating_timestamp_utc LIKE '%2020-04-%'; user is a trial list refers to user_trialist = 1;",
        "SQL": "SELECT T1.movie_title FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T2.user_trialist = 1 AND T2.rating_timestamp_utc LIKE '2020-04%'"
    },
    {
        "question": "List ther users who gave the worst rating for movie 'Love Will Tear Us Apart'.",
        "evidence": "worst rating refers to rating_score = 1;",
        "SQL": "SELECT T1.user_id FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'Love Will Tear Us Apart' AND T1.rating_score = 1"
    },
    {
        "question": "List all movies with the best rating score. State the movie title and number of Mubi user who loves the movie.",
        "evidence": "best rating score refers to rating_score = 5; number of Mubi user who loves the movie refers to movie_popularity;",
        "SQL": "SELECT DISTINCT T2.movie_title, T2.movie_popularity FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_score = 5"
    },
    {
        "question": "For all ratings which are rated in year 2020, name the movies which has the rating scored 4 and above.",
        "evidence": "ratings in year 2020 refers to rating_timestamp_utc like '%2020%'; rating_score > = 4;",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE CAST(SUBSTR(T1.rating_timestamp_utc, 1, 4) AS INTEGER) = 2020 AND CAST(SUBSTR(T1.rating_timestamp_utc, 6, 2) AS INTEGER) > 4"
    },
    {
        "question": "For all movies where users left a critic, find the movie name, user, rating and critics comments from the user.",
        "evidence": "movies where users left a critic refers to critic IS NOT NULL; critic comments refers to critic;",
        "SQL": "SELECT T2.movie_title, T1.user_id, T1.rating_score, T1.critic FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.critic IS NOT NULL"
    },
    {
        "question": "For movie titled 'Welcome to the Dollhouse', how many percentage of the ratings were rated with highest score.",
        "evidence": "rated with highest score refers to rating_score = 5; percentage = MULTIPLY(DIVIDE(SUM(rating_score = 5), COUNT(rating_score)), 100)",
        "SQL": "SELECT CAST(SUM(CASE WHEN T2.rating_score = 5 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_title = 'Welcome to the Dollhouse'"
    },
    {
        "question": "What is the percentage of rated movies were released in year 2021?",
        "evidence": "percentage = DIVIDE(SUM(movie_release_year = 2021), COUNT(rating_id)) as percent; movies released in year 2021 refers to movie_release_year = 2021;",
        "SQL": "SELECT CAST(SUM(CASE WHEN T1.movie_release_year = 2021 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id"
    },
    {
        "question": "Who is the director of the movie Sex, Drink and Bloodshed?",
        "evidence": "Sex, Drink and Bloodshed refers to movie title = 'Sex, Drink and Bloodshed';",
        "SQL": "SELECT director_name FROM movies WHERE movie_title = 'Sex, Drink and Bloodshed'"
    },
    {

        "question": "What is the name of the most followed list?",
        "evidence": "most followed list refers to MAX(list_followers);",
        "SQL": "SELECT list_title FROM lists ORDER BY list_followers DESC LIMIT 1"
    },
    {
        "question": "What are the URL to the list page on Mubi of the lists with followers between 1-2 and whose last update timestamp was on 2012?",
        "evidence": "URL to the list page on Mubi refers to list_url; list_followers = 1 OR list_followers = 2; last update timestamp was on 2012 refers to list_update_timestamp_utc BETWEEN '2012-1-1' AND '2012-12-31';",
        "SQL": "SELECT list_url FROM lists WHERE list_update_timestamp_utc LIKE '2012%' AND list_followers BETWEEN 1 AND 2 ORDER BY list_update_timestamp_utc DESC LIMIT 1"
    },
    {
        "question": "What is the list ID that was first created by user 85981819?",
        "evidence": "first created list refers to oldest list_creation_date_utc;",
        "SQL": "SELECT list_id FROM lists_users WHERE user_id = 85981819 ORDER BY list_creation_date_utc ASC LIMIT 1"
    },
    {
        "question": "For movie id 1269, how many users, who was a paying subscriber and was eligible for trial when he rated the movie, gave the movie a rating score of less than or equal to 2?",
        "evidence": "paying subscriber refers to user_has_payment_method = 1; eligible for trial refers to user_eligible_for_trial = 1; rating_score< = 2;",
        "SQL": "SELECT COUNT(*) FROM ratings WHERE movie_id = 1269 AND rating_score <= 2 AND user_eligible_for_trial = 1 AND user_has_payment_method = 1"
    },
    {
        "question": "What are the movie popularity of the movies released in 2021 that were directed by Steven Spielberg? List the names of the movies and their corresponding popularity.",
        "evidence": "movie released in 2021 refers to movie_release_year = 2021; popularity refers to movie_popularity;",
        "SQL": "SELECT movie_title, movie_popularity FROM movies WHERE movie_release_year = 2021 AND director_name = 'Steven Spielberg'"
    },
    {
        "question": "When was the first movie released and who directed it?",
        "evidence": "first movie refers to oldest movie_release_year;",
        "SQL": "SELECT movie_release_year, director_name FROM movies WHERE movie_release_year IS NOT NULL ORDER BY movie_release_year ASC LIMIT 1"
    },
    {
        "question": "What is the user ID of the user, who was a subscriber when he created the list, who created a list for 10 consecutive years? If there are multiple users, indicate each of their user IDs.",
        "evidence": "user was a subscriber when he created the list refers to user_subscriber = 1; user who created a list for 10 consecutive years refers to user_id with list_creation_date_utc for 10 succeeding years;",
        "SQL": "SELECT user_id FROM lists_users WHERE user_subscriber = 1 GROUP BY user_id HAVING MAX(SUBSTR(list_creation_date_utc, 1, 4)) - MIN(SUBSTR(list_creation_date_utc, 1, 4)) >= 10"
    },
    {
        "question": "How many users gave \"Pavee Lackeen: The Traveller Girl\" movie a rating score of 4?",
        "evidence": "FALSE;",
        "SQL": "SELECT COUNT(T2.user_id) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_title = 'Pavee Lackeen: The Traveller Girl' AND T2.rating_score = 4"
    },
    {
        "question": "Was the user who created the \"World War 2 and Kids\" list eligible for trial when he created the list? Indicate how many followers does the said list has.",
        "evidence": "user was eligible for trial when he created the list refers to user_eligible_for_trial = 1; number of followers a list have refers to list_followers;",
        "SQL": "SELECT T2.user_eligible_for_trial, T1.list_followers FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.user_id = T1.user_id AND T1.list_id = T2.list_id WHERE T1.list_title = 'World War 2 and Kids'"
    },
    {
        "question": "Which year was the third movie directed by Quentin Tarantino released? Indicate the user ids of the user who gave it a rating score of 4.",
        "evidence": "third movie refers to third movie that has oldest movie_release_year;",
        "SQL": "SELECT T2.movie_release_year, T1.user_id FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_id = ( SELECT movie_id FROM movies WHERE director_name = 'Quentin Tarantino' ORDER BY movie_release_year ASC LIMIT 2, 1 ) AND T1.rating_score = 4"
    },
    {
        "question": "What is the URL to the movie director page on Mubi of the director whose movie was critic by user 2452551 and was given 39 likes?",
        "evidence": "URL to the movie director page on Mubi refers to director_url; likes refers to critic_likes; critic_likes = 39;",
        "SQL": "SELECT T2.director_url FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.user_id = 2452551 AND T1.critic_likes = 39"
    },
    {
        "question": "What is the average rating score of the movie \"When Will I Be Loved\" and who was its director?",
        "evidence": "average rating score = AVG(rating_score);",
        "SQL": "SELECT AVG(T1.rating_score), T2.director_name FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'When Will I Be Loved'"
    },
    {
        "question": "How many movies were added to the list with the most number of movies? Indicate whether the user was a paying subscriber or not when he created the list.",
        "evidence": "list with the most number of movies refers to MAX(list_movie_number); user_has_payment_method = 1 means the user was a paying subscriber when he created the list; user_has_payment_method = 0 means the user was not a paying subscriber when he created the list;",
        "SQL": "SELECT T1.list_movie_number, T2.user_has_payment_method FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id ORDER BY T1.list_movie_number DESC LIMIT 1"
    },
    {
        "question": "What is the name of the movie whose critic received the highest number of likes related to the critic made by the user rating the movie?",
        "evidence": "number of likes received refers to critic likes; received the highest number of likes refers to MAX(critic_likes);",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id ORDER BY T1.critic_likes DESC LIMIT 1"
    },
    {
        "question": "How much is the popularity of the movie that has the highest popularity between 1920 to 1929 and when did the movie received its first rating score of 1 from the users who were a paying subscriber when they rated the movie ?",
        "evidence": "movie with highest popularity refers to MAX(movie_popularity); movie_release_year BETWEEN 1920 AND 1929; when the movie received its first rating score of 1 refers to oldest date in rating_timestamp_utc where rating score = 1; user was a paying subscriber when they rated the movie refers to user_has_payment_method = 1;",
        "SQL": "SELECT MAX(T2.movie_popularity), MIN(T1.rating_timestamp_utc) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_release_year BETWEEN 1920 AND 1929 AND T1.rating_score = 1 AND T1.user_has_payment_method = 1"
    },
    {
        "question": "How many movies directed by Francis Ford Coppola have a popularity of more than 1,000? Indicate what is the highest amount of likes that each critic per movie has received, if there's any.",
        "evidence": "Francis Ford Coppola refers to director_name; popularity of more than 1,000 refers to movie_popularity >1000;highest amount of likes that each critic per movie has received refers to MAX(critic_likes)",
        "SQL": "SELECT COUNT(T2.movie_title), T1.critic FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.director_name = 'Francis Ford Coppola' AND T2.movie_popularity > 1000"
    },
    {
        "question": "What is the URL to the user profile image on Mubi of the user who gave the movie id of 1103 a 5 ratinng score on 4/19/2020?",
        "evidence": "URL to the user profile image on Mubi\u00a0 refers to user_avatar_image_url;\u00a0 4/19/2020 refers to rating_date_utc",
        "SQL": "SELECT T2.user_avatar_image_url FROM ratings AS T1 INNER JOIN ratings_users AS T2 ON T1.user_id = T2.user_id WHERE T2.user_id = 1103 AND rating_score = 5 AND T2.rating_date_utc = '2020-04-19'"
    },
    {
        "question": "Among the lists created by user 4208563, which one has the highest number of followers? Indicate how many followers it has and whether the user was a subscriber or not when he created the list.",
        "evidence": "User 4208563 refers to user_id;highest number of followers refers to MAX(list_followers); user_subscriber = 1 means that the user was a subscriber when he created the list; user_subscriber = 0 means the user was not a subscriber when he created the list (to replace)",
        "SQL": "SELECT T1.list_followers, T2.user_subscriber = 1 FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.user_id = T2.user_id AND T2.list_id = T2.list_id WHERE T2.user_id = 4208563 ORDER BY T1.list_followers DESC LIMIT 1"
    },
    {
        "question": "Which year has the least number of movies that was released and what is the title of the movie in that year that has the highest number of rating score of 1?",
        "evidence": "least number of movies refers to MIN(movie_release_year); highest rating score refers to MAX(SUM(movie_id) where rating_score = '1')\n\n",
        "SQL": "SELECT DISTINCT T1.movie_release_year, T1.movie_title FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_release_year = ( SELECT movie_release_year FROM movies GROUP BY movie_release_year ORDER BY COUNT(movie_id) DESC LIMIT 1 ) AND T2.rating_score = 1"
    },
    {
        "question": "How many users, who were a paying subscriber when they rated the movie, gave the movie that was released in 1924 and directed by Erich von Stroheim a rating score of 5?",
        "evidence": "Directed by Buster Keaton refers to director_name; released in 1924 refers to movie_release_year = 1924; paying subscriber refers to user_has_payment_method = 1\n\n",
        "SQL": "SELECT COUNT(T2.user_id) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_release_year = 1924 AND T1.director_name = 'Erich von Stroheim' AND T2.rating_score = 5 AND T2.user_has_payment_method = 1"
    },
    {
        "question": "What is the average number of movies added to the lists of user 8516503? Give the user profile image URL on Mubi.",
        "evidence": "user profile image URL refers to user_avatar_image_url; user 8516503 refers to user_id; Average refers to AVG(list_movie_number where user_id = 8516503)\n\n",
        "SQL": "SELECT AVG(T1.list_movie_number), T2.user_avatar_image_url FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id AND T1.user_id = T2.user_id WHERE T2.user_id = 8516503"
    },
    {
        "question": "How many users rated the movie \"The Magnificent Ambersons\" gave a rating score of no more than 2? List all the URL to the rating on Mubi.",
        "evidence": "The Magnificent Ambersons refers to movie_title; rating score of no more than 2 refers to rating_score<2; URL to rating refers to rating_url",
        "SQL": "SELECT COUNT(T2.user_id), T2.rating_url FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_title = 'The Magnificent Ambersons' AND T2.rating_score <= 2"
    },
    {
        "question": "How many users who created a list in the February of 2016 were eligible for trial when they created the list? Indicate the user id of the user who has the most number of followers in his list in February of 2016.",
        "evidence": "created a list in the February of 2016 refer to list_creation_date_utc BETWEEN 2/1/2016 and 2/29/2016; eligible for trial refers to user_eligible_for_trial = 1;\n",
        "SQL": "SELECT T1.list_followers FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.user_id = T2.user_id AND T1.list_id = T2.list_id WHERE T2.list_creation_date_utc BETWEEN '2016-02-01' AND '2016-02-29' AND T2.user_eligible_for_trial = 1"
    },
    {
        "question": "What is the URL to the rating on Mubi of the Riff-Raff movie that was given the highest rating score by user 22030372?",
        "evidence": "URL refer to rating_url; user 22030372 refer to user_id",
        "SQL": "SELECT T2.rating_url FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T2.user_id = 22030372 AND T2.rating_score = 5 AND T1.movie_title = 'Riff-Raff'"
    },
    {
        "question": "How many directors have directed atleast 10 movies between 1960 to 1985? Indicate the name of the movie in those years of each director that received the highest amount of 5 rating score.",
        "evidence": "directed at least 10 movies refers to count(direct_name)>10; 1960 to 1985 refer to movie_release_year\n",
        "SQL": "SELECT T2.director_name FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_release_year BETWEEN 1960 AND 1985 GROUP BY T2.director_name HAVING COUNT(T2.movie_id) > 10"
    },
    {
        "question": "How many users, who were not a a trialist when they rated the movie, gave the movie \"The South\" a rating score of not more than 2?",
        "evidence": "not a trialist refer to user_trialist = 0; rating score of not more than 2 refer to rating_score <2; The South refers to movie_title\n",
        "SQL": "SELECT COUNT(T2.user_id) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T2.user_trialist = 0 AND T2.rating_score <= 2 AND T1.movie_title = 'The South'"
    },
    {
        "question": "How many likes did the critic of the movie \"Apocalypse Now\" received after giving the movie a rating score of 5?",
        "evidence": "Apocalypse Now refer to movie_title; rating score refer to rating_score = '5';likes received refers to critic_likes\n",
        "SQL": "SELECT T2.critic_likes FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T2.user_trialist = 0 AND T2.rating_score = 5 AND T1.movie_title = 'Apocalypse Now'"
    },
    {
        "question": "What is the average rating score of the movie \"The Crowd\" and who was its director?",
        "evidence": "director refer to director_name; The Crowd refer to movie_title; Average refer to AVG(rating_score)",
        "SQL": "SELECT AVG(T2.rating_score), T1.director_name FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_title = 'The Crowd'"
    },
    {
        "question": "When was the first movie of the director who directed the highest number of movies released and what is the user id of the user who received the highest number of comments related to the critic made by the user rating the movie?",
        "evidence": "comments refer to critic_comments",
        "SQL": "SELECT MIN(movie_release_year) FROM movies WHERE director_name = ( SELECT T2.director_name FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_release_year BETWEEN 1960 AND 1985 GROUP BY T2.director_name ORDER BY COUNT(T2.director_name) DESC LIMIT 1 )"
    },
    {
        "question": "How many movies have a popularity of more than 400 but less than 500? Indicate the name of the movies and the highest rating score each movie has received.",
        "evidence": "popularity of more than 400 but less than 500 refers to movie_popularity BETWEEN 400 AND 500; highest rating score refer to MAX(rating_score)\n\n",
        "SQL": "SELECT T1.movie_title, MAX(T2.rating_score) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T1.movie_popularity BETWEEN 400 AND 500 GROUP BY T1.movie_title"
    },
    {
        "question": "What is the URL to the rating on Mubi made by user 45579900 for the movie \"The Vertical Ray of the Sun\" that received 20 likes?",
        "evidence": "URL refer to rating_url; 20 likes refer to critic_likes = \u201920\u2019; user 45579900 refer to user_id",
        "SQL": "SELECT T2.rating_url FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T2.user_id = 45579900 AND T1.movie_title = 'The Vertical Ray of the Sun' AND T2.critic_likes = 20"
    },
    {
        "question": "What is the average popularity of each movie that was directed by Christopher Nolan? Indicate which movie directed by him has received the highest number of 5 rating scores.",
        "evidence": "5 rating scores refer to rating_score; Christopher Nolan refer to director_name; average popularity of each movie refer to AVG(movie_popularity where director_name = 'Christopher Nolan')",
        "SQL": "SELECT AVG(T2.movie_popularity) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.director_name = 'Christopher Nolan'"
    },
    {
        "question": "What are the names of the movie that was rated by the user between 1/1/2013 to 12/31/2013 by the user who created the list \"100 Greatest Living American Filmmakers\"? Calculate for the average rating score of those movies in 2013.",
        "evidence": "Between 1/1/2013 to 12/31/2013 refer to rating_timestamp_utc; 100 Greatest Living American Filmmakers refer to list_title; average rating score refer to DIVIDE( ADD(rating_score where rating_timestamp_utc = '1/1/2013-12/31/2013'), COUNT(rating_timestamp_utc = '1/1/2013-12/31/2013'))",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id INNER JOIN lists AS T3 ON T3.user_id = T1.user_id WHERE T1.rating_timestamp_utc BETWEEN '2013-01-01' AND '2013-12-31' AND T3.list_title = '100 Greatest Living American Filmmakers'"
    },
    {
        "question": "What is the average rating score of the 'Pavee Lackeen: The Traveller Girl' movie and what year was it released?",
        "evidence": "year it was released refers to movie_release_year; average rating score refers to AVG(rating_score where movie_title = 'Final Destination 6'); Final Destination 6 refers to movie_title",
        "SQL": "SELECT AVG(T1.rating_score), T2.movie_release_year FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'Pavee Lackeen: The Traveller Girl'"
    },
    {
        "question": "How many movie lists were still updated 10 years after it was created?",
        "evidence": "updated 10 years after it was created refers to list_update_timestamp_utc > (list_creation_timestamp_utc+10);",
        "SQL": "SELECT COUNT(*) FROM lists WHERE SUBSTR(list_update_timestamp_utc, 1, 4) - SUBSTR(list_creation_timestamp_utc, 1, 4) > 10"
    },
    {
        "question": "What's the description for the movie list \"Short and pretty damn sweet\"?",
        "evidence": "Short and pretty damn sweet is list_title; description refers to list_description;",
        "SQL": "SELECT list_description FROM lists WHERE list_title = 'Short and pretty damn sweet'"
    },
    {
        "question": "Where can I find the movie list \"Short and pretty damn sweet\"?",
        "evidence": "Short and pretty damn sweet is list_title; location of the movie refers to list_url;",
        "SQL": "SELECT list_url FROM lists WHERE list_title = 'Short and pretty damn sweet'"
    },
    {
        "question": "Among the movie lists created after 2010/1/1, how many of them have over 200 followers?",
        "evidence": "created after 2010/1/1 refers to list_update_timestamp_utc>'2010/1/1'; over 200 followers refers to list_followers>200;",
        "SQL": "SELECT COUNT(*) FROM lists WHERE list_followers > 200 AND list_update_timestamp_utc > '2010-01-01'"
    },
    {
        "question": "How many movie lists were created by user 83373278 when he or she was a subscriber?",
        "evidence": "the user was a subscriber when he created the list refers to user_subscriber = 1; user 83373278 refers to user_id = 83373278;",
        "SQL": "SELECT COUNT(*) FROM lists_users WHERE user_id = 83373278 AND user_subscriber = 1"
    },
    {
        "question": "In which year was the movie \"La Antena\" released?",
        "evidence": "movie La Antena refers to movie_title = 'La Antena'; which year refers to movie_release_year;",
        "SQL": "SELECT movie_release_year FROM movies WHERE movie_title = 'La Antena'"
    },
    {
        "question": "Please give me the url of the movie \"La Antena\".",
        "evidence": "movie La Antena refers to movie_title = 'La Antena'; url refers to movie_url;",
        "SQL": "SELECT movie_url FROM movies WHERE movie_title = 'La Antena'"
    },
    {
        "question": "Which movie is more popular, \"The General\" or \"Il grido\"?",
        "evidence": "The General and Il grido are movie_title; more popular movie refers to higher (movie_popularity);",
        "SQL": "SELECT movie_title FROM movies WHERE movie_title = 'The General' OR movie_title = 'Il grido' ORDER BY movie_popularity DESC LIMIT 1"
    },
    {
        "question": "How many movies registered on Mubi are directed by Hong Sang-soo?",
        "evidence": "Hong Sang-soo is the name of director;",
        "SQL": "SELECT COUNT(movie_id) FROM movies WHERE director_name = 'Hong Sang-soo'"
    },
    {
        "question": "Was the user who created the list \"250 Favourite Films\" a trialist when he or she created the list?",
        "evidence": "the user was a trialist when he created the list refers to user_trailist = 1; 250 Favourite Films is list_title;",
        "SQL": "SELECT T2.user_trialist FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id AND T1.user_id = T2.user_id WHERE T1.list_title = '250 Favourite Films'"
    },
    {
        "question": "Please list the titles of the movie lists user 32172230 created when he or she was eligible for trial.",
        "evidence": "the user was eligible for trail when he created the list refers to user_eligile_for_trail = 1; user 32172230 refers to user_id = 32172230;",
        "SQL": "SELECT T1.list_title FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id AND T1.user_id = T2.user_id WHERE T1.user_id = 32172230 AND T2.user_eligible_for_trial = 1"
    },
    {
        "question": "How many movie lists with over 100 movies had user 85981819 created when he or she was a paying subscriber?",
        "evidence": "the user was a paying subscriber when he created the list refers to user_has_payment_method = 1;\u00a0 movie lists with over 100 refers to list_movie_number >100;\u00a0 user 85981819 refers to user_id = 85981819;",
        "SQL": "SELECT COUNT(*) FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id AND T1.user_id = T2.user_id WHERE T1.user_id = 85981819 AND T1.list_movie_number > 100 AND T2.user_has_payment_method = 1"
    },
    {
        "question": "What's the description of user 85981819's movie list with the most followers?",
        "evidence": "user 85981819 refers to user_id = 85981819; most followers refers to Max(list_followers); description refers to list_descriptions;",
        "SQL": "SELECT T1.list_description FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id AND T1.user_id = T2.user_id WHERE T1.user_id = 85981819 ORDER BY T1.list_followers DESC LIMIT 1"
    },
    {
        "question": "When did the creator of the list \"250 Favourite Films\" last updated a movie list?",
        "evidence": "250 Favourite Films refers to list_title; last update refers to list_update_date_utc;",
        "SQL": "SELECT T2.list_update_date_utc FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id AND T1.user_id = T2.user_id WHERE T1.list_title = '250 Favourite Films' ORDER BY T2.list_update_date_utc DESC LIMIT 1"
    },
    {
        "question": "What's the avatar image of the user who created the movie list \"250 Favourite Films\"?",
        "evidence": "250 Favourite Films refers to list_title; avatar image refers to user_avatar_image_url;",
        "SQL": "SELECT T2.user_avatar_image_url FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id AND T1.user_id = T2.user_id WHERE T1.list_title = '250 Favourite Films'"
    },
    {
        "question": "How many more movie lists were created by the user who created the movie list \"250 Favourite Films\"?",
        "evidence": "250 Favourite Films refers to list_title;",
        "SQL": "SELECT COUNT(list_id) FROM lists_users WHERE user_id = ( SELECT user_id FROM lists WHERE list_title = '250 Favourite Films' )"
    },
    {
        "question": "How many users liked the movie \"A Way of Life\" to the highest extent?",
        "evidence": "like the movie highest to the extent refers to rating_score = 5; A Way of Life refers to movie_title;",
        "SQL": "SELECT COUNT(T1.user_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'A Way of Life' AND T1.rating_score = 5"
    },
    {
        "question": "Please list all the critics made by the user rating the movie \"A Way of Life\".",
        "evidence": "A Way of Life refers to movie_title;",
        "SQL": "SELECT T1.critic FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'A Way of Life'"
    },
    {
        "question": "How many critics of the movie \"Imitation of Life\" got more than 1 like?",
        "evidence": "Imitation of Life refers to movie_title; critics got more than 1 like refers to critic_likes >1;",
        "SQL": "SELECT COUNT(*) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'Imitation of Life' AND T1.critic_likes > 1"
    },
    {
        "question": "Which user made a critic for the film \"When Will I Be Loved\" and got 2 comments for the critic?",
        "evidence": "When Will I Be Loved refers to movie_title;\u00a0 2 comments for the critic refers to critic_comments = 2;",
        "SQL": "SELECT T1.user_id FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'When Will I Be Loved' AND T1.critic_comments = 2"
    },
    {
        "question": "When did user 39115684 rate the movie \"A Way of Life\"?",
        "evidence": "A Way of Life' refers to movie_title; user 39115684 refers to userid = 39115684;\u00a0 when the user rate refers to rating_timestamp_utc;",
        "SQL": "SELECT T1.rating_score FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'A Way of Life' AND T1.user_id = 39115684"
    },
    {
        "question": "What's the url of user 39115684's rating on the movie 'When Will I Be Loved'?",
        "evidence": "A Way of Life refers to movie_title; user 39115684 refers to userid = 39115684;\u00a0 url refers to rating_url;",
        "SQL": "SELECT T1.rating_url FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'A Way of Life' AND T1.user_id = 39115684"
    },
    {
        "question": "Was user 39115684 a trialist when he or she rated the movie \"A Way of Life\"?",
        "evidence": "A Way of Life' refers to movie_title; user 39115684 refers to userid = 39115684;\u00a0 the user was a trialist when he rated the movie refers to user_trialist = 1;",
        "SQL": "SELECT T1.user_trialist FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'A Way of Life' AND T1.user_id = 39115684"
    },
    {
        "question": "How many users were trialists when they rated the movie \"A Way of Life\"?",
        "evidence": "A Way of Life' refers to movie_title; the user was a trialist when he rated the movie refers to user_trialist = 1;",
        "SQL": "SELECT COUNT(T1.user_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'When Will I Be Loved' AND T1.user_trialist = 1"
    },
    {
        "question": "Please list all the links to the ratings on the movie \"A Way of Life\" with a critic.",
        "evidence": "A Way of Life' refers to movie_title; with a critic refers to critic is not null, links to the ratings refers to rating_url;",
        "SQL": "SELECT T1.rating_url FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'A Way of Life' AND T1.critic IS NOT NULL"
    },
    {
        "question": "How many users have rated the most popular movie?",
        "evidence": "most popular refers to Max(movie_popularity);",
        "SQL": "SELECT COUNT(rating_id) FROM ratings WHERE movie_id = ( SELECT movie_id FROM movies ORDER BY movie_popularity DESC LIMIT 1 )"
    },
    {
        "question": "User 58149469's critic on which film got 1 like and 2 comments?",
        "evidence": "user 58149469 refers to user_id = 58149469; critic with 1 like refers to critic_likes = 1; critic with 2 comments refers to critic_comments = 2;",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.user_id = 58149469 AND T1.critic_likes = 1 AND T1.critic_comments = 2"
    },
    {
        "question": "Among the users who are trailists when rating the movie \"When Will I Be Loved\", how many of them have rated \"1\" on the movie?",
        "evidence": "When Will I Be Loved refers to movie_title; the user was a trialist when he rated the movie refers to user_trialist = 1;rated 1 on the movie refers to rating_score = 1;",
        "SQL": "SELECT COUNT(T1.user_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'When Will I Be Loved' AND T1.rating_score = 1 AND T1.user_trialist = 1"
    },
    {
        "question": "How many ratings on the movie \"A Way of Life\" are made after the year 2011?",
        "evidence": "A Way of Life' is movie_title; rating after the year 2011 refers to rating_timestamp_utc > '2011';",
        "SQL": "SELECT COUNT(T1.rating_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'A Way of Life' AND T1.rating_timestamp_utc >= '2012-01-01'"
    },
    {
        "question": "What's of rating on the movie \"Innocence Unprotected\" by the user who created the movie list \"250 Favourite Films\"?",
        "evidence": "Innocence Unprotected' is movie_title; '250 Favourite Films' is list_title; rating refers to rating_score;",
        "SQL": "SELECT T1.rating_score FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id INNER JOIN lists AS T3 ON T3.user_id = T1.user_id WHERE T2.movie_title = 'Innocence Unprotected' AND T3.list_title = '250 Favourite Films'"
    },
    {
        "question": "Please list the movies rated by the user who created the movie list \"250 Favourite Films\".",
        "evidence": "250 Favourite Films' is list_title; movies refers to movie_title;",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id INNER JOIN lists AS T3 ON T3.user_id = T1.user_id WHERE T3.list_title = '250 Favourite Films'"
    },
    {
        "question": "What's the average rating score of the movie \"A Way of Life\"?",
        "evidence": "A Way of Life' is movie_title; average rating score = Divide (Sum(rating_score), Count(rating_id));",
        "SQL": "SELECT AVG(T1.rating_score) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'A Way of Life'"
    },
    {
        "question": "What's the percentage of the users who have rated \"1\" on the movie \"When Will I Be Loved\"?",
        "evidence": "When Will I Be Loved' is movie_title; rated 1 refers to rating_score = 1; percentage = Divide(Count(rating_id where rating_score = 1),Count(rating_id)) *100;",
        "SQL": "SELECT CAST(SUM(CASE WHEN T1.rating_score = 1 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'When Will I Be Loved'"
    },
    {
        "question": "How much higher is the average rating score of the movie \"Innocence Unprotected\" than the movie \"When Will I Be Loved\"?",
        "evidence": "Innocence Unprotected' and 'When Will I Be Loved' are movie_title; Average rating score = Divide(Sum(rating_score), Count(rating_id));",
        "SQL": "SELECT SUM(CASE WHEN T2.movie_title = 'Innocence Unprotected' THEN T1.rating_score ELSE 0 END) / SUM(CASE WHEN T2.movie_title = 'Innocence Unprotected' THEN 1 ELSE 0 END) - SUM(CASE WHEN T2.movie_title = 'When Will I Be Loved' THEN T1.rating_score ELSE 0 END) / SUM(CASE WHEN T2.movie_title = 'When Will I Be Loved' THEN 1 ELSE 0 END) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id"
    },
    {
        "question": "Who was the director of the movie \"Tokyo Eyes\"\uff1f",
        "evidence": "Tokyo Eyes' is movie_title, director refers to director_name;",
        "SQL": "SELECT director_name FROM movies WHERE movie_title = 'Tokyo Eyes'"
    },
    {
        "question": "How many films were released in 2007?",
        "evidence": "film released in 2007 refers to movie_release_year = 2007; film refers to movie",
        "SQL": "SELECT COUNT(*) FROM movies WHERE movie_release_year = 2007"
    },
    {
        "question": "Which of the films released in 2006 was the most popular among Mubi users?",
        "evidence": "released in 2006 refers to movie_release_year = 2006; most popular refers to Max(movie_popularity); film refers to movie;",
        "SQL": "SELECT movie_title FROM movies WHERE movie_release_year = 2006 ORDER BY movie_popularity DESC LIMIT 1"
    },
    {
        "question": "How many films did \u00c5ke Sandgren direct?",
        "evidence": "Ake Sandgren is the director name;\u00a0 film refers to movie",
        "SQL": "SELECT COUNT(movie_title) FROM movies WHERE director_name = '\u00c5ke Sandgren'"
    },
    {
        "question": "Which of the films directed by \u00c1lex de la Iclesia is the most popular among Mubi users?",
        "evidence": "Alex de la Iclesia is the director name; the most popular refers to Max(movie_popularity); films refers to movies;",
        "SQL": "SELECT movie_title FROM movies WHERE director_name = '\u00c5ke Sandgren' ORDER BY movie_popularity DESC LIMIT 1"
    },
    {
        "question": "When was the movie Cops released?",
        "evidence": "Cops' is movie_title; released refers to movie_release_year;",
        "SQL": "SELECT movie_release_year FROM movies WHERE movie_title = 'Cops'"
    },
    {
        "question": "Please list the id of the director of the movie \"It's Winter\".",
        "evidence": "It's Winter' is movie_title;",
        "SQL": "SELECT director_id FROM movies WHERE movie_title = 'It''s Winter'"
    },
    {
        "question": "Please provide the ID of the user with the most followers on the list.",
        "evidence": "most followers refers to Max(list_followers);",
        "SQL": "SELECT user_id FROM lists ORDER BY list_followers DESC LIMIT 1"
    },
    {
        "question": "Please provide the title of the list with the most comments on the list.",
        "evidence": "the most comments refers to Max(list_comments);",
        "SQL": "SELECT list_title FROM lists GROUP BY list_title ORDER BY COUNT(list_comments) DESC LIMIT 1"
    },
    {
        "question": "Which of the film released in 2008 scored the highest?",
        "evidence": "film released in 2008 refers to movie_release_year = 2008; scored the highest refers to Max(rating_score); film refers to movie;",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_release_year = 2008 ORDER BY T1.rating_score DESC LIMIT 1"
    },
    {
        "question": "Please list the names of the top three movies in the number of likes related to the critic made by the user rating the movie.",
        "evidence": "likes related to the critic made by the user rating the movie refers to critic_likes; top refers to Max(critic_likes);",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id ORDER BY T1.critic_likes DESC LIMIT 3"
    },
    {
        "question": "How many users have more than 100 followers in the list created by users in 2009?",
        "evidence": "more than 100 followers refers to list_followers >100;\u00a0 list created by the user in 2009 refers to list_creation_date_utc = '2009';",
        "SQL": "SELECT COUNT(T1.user_id) FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_followers > 100 AND T1.list_creation_date_utc LIKE '2009%'"
    },
    {
        "question": "How many users in Mubi give the movie \"White Night Wedding for 5\"?",
        "evidence": "White Night Wedding' is movie_title; for 5 refers to rating_score = 5;",
        "SQL": "SELECT COUNT(T1.user_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_score = 5 AND T2.movie_title = 'White Night Wedding'"
    },
    {
        "question": "What's the cover image of the user who created the movie list 'Georgia related films'?",
        "evidence": "Play it cool' is list_title; cover image of user refers to user_cover_image_url;",
        "SQL": "SELECT T1.user_cover_image_url FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_title LIKE 'Georgia related films'"
    },
    {
        "question": "How many followers does the list created by the user whose user_avatar_image_url is https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214 have?",
        "evidence": "followers refers to list_followers;",
        "SQL": "SELECT SUM(T2.list_followers) FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T1.user_avatar_image_url = 'https://assets.mubicdn.net/images/avatars/74983/images-w150.jpg?1523895214'"
    },
    {
        "question": "Please list the names of the movies that user 94978 scored as 5.",
        "evidence": "user 94978 refers to user_id = 94978; scored as 5 refers to rating_score = 5;",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_score = 5 AND T1.user_id = 94978"
    },
    {
        "question": "Please list the names of the films released in 2003 among the films scored by user 2941 .",
        "evidence": "released in 2003 refers to movie_release_year = 2003; user 2941 refers to user_id = 2941; film refers to movie;",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_release_year = 2003 AND T1.user_id = 2941"
    },
    {
        "question": "How many users were not trialists when they rated the movie \"Patti Smith: Dream of Life\"?",
        "evidence": "Patti Smith: Dream of Life' is movie_title; the user was not a trialist when he created the list refers to user_trialist = 0;",
        "SQL": "SELECT COUNT(T1.user_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'Patti Smith: Dream of Life' AND T1.user_trialist = 0"
    },
    {
        "question": "Which movie has the highest average score in Mubi?",
        "evidence": "Highest average score refers to Max(Avg(rating_score));",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id GROUP BY T2.movie_title ORDER BY SUM(T1.rating_score) / COUNT(T1.rating_id) DESC LIMIT 1"
    },
    {
        "question": "Please list the names of the top three movies in the number comments related to the critic made by the user rating the movie.",
        "evidence": "number of comments related to the critic made by the user rating the movie refers to critic_comments; top movie refers to Max(critic_comments);",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id ORDER BY T1.critic_comments DESC LIMIT 3"
    },
    {
        "question": "What was the title of the first list created by a user 85981819? And please provide the user_avatar_image_url.",
        "evidence": "user 85981819 refers to user_id = 85981819;\u00a0 first list created refers to Min (list_creation_date_utc);",
        "SQL": "SELECT T2.list_title, T1.user_avatar_image_url FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T1.user_id = 85981819 ORDER BY T2.list_creation_timestamp_utc LIMIT 1"
    },
    {
        "question": "Please list the names of the movies that have been rated the most times in 2020.",
        "evidence": "in 2020 refers to rating_timestamp_utc = '2020%'; rated the most times refers to Max(Count(movie_title));",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_timestamp_utc LIKE '2020%' GROUP BY T2.movie_title ORDER BY COUNT(T2.movie_title) DESC LIMIT 1"
    },
    {
        "question": "What is the average score for the movie Versailles Rive-Gauche?",
        "evidence": "Versailles Rive-Gauche' is movie_title; average score refers to Avg(rating_score);",
        "SQL": "SELECT AVG(T1.rating_score) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title LIKE 'Versailles Rive-Gauche'"
    },
    {
        "question": "Which film rated by user 59988436 that received 21 comments?",
        "evidence": "user 59988436 refers to user_id = 59988436; received 21 comments refers to critic_comments = 21; film refers to movie;",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.user_id = 59988436 AND T1.critic_comments = 21"
    },
    {
        "question": "Please list the names of the movies that received more than 20 likes?",
        "evidence": "received more than 20 likes refers to critic_likes>20;",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.critic_likes > 20"
    },
    {
        "question": "What is the average score of the movie \"The Fall of Berlin\" in 2019?",
        "evidence": "The Fall of Berlin' is movie_title; in 2019 refers to rating_timestamp_utc = 2019; Average score refers to Avg(rating_score);",
        "SQL": "SELECT SUM(T1.rating_score) / COUNT(T1.rating_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_timestamp_utc LIKE '2019%' AND T2.movie_title LIKE 'The Fall of Berlin'"
    },
    {
        "question": "What percentage of users rated the movie \"Patti Smith: Dream of Life\" by more than 3?",
        "evidence": "Patti Smith: Dream of Life' is movie_title; more than 3 refers to rating_score >3; percentage = Divide(Count(rating_score where rating_score >3), Count(rating_score))*100",
        "SQL": "SELECT CAST(SUM(CASE WHEN T1.rating_score > 3 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.rating_score) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title LIKE 'Patti Smith: Dream of Life'"
    },
    {
        "question": "Which of the film directed by director Abbas Kiarostami has the highest average score?",
        "evidence": "Abbas Kiarostami' is director_name; the highest Average score refers to Max(Avg(rating_score));",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.director_name = 'Abbas Kiarostami' GROUP BY T2.movie_title ORDER BY SUM(T1.rating_score) / COUNT(T1.rating_id) DESC LIMIT 1"
    },
    {
        "question": "Which year had the most released films?",
        "evidence": "year refers to movie_release_year; most release films refers to MAX(COUNT(movie_id))\n\n",
        "SQL": "SELECT movie_release_year FROM movies GROUP BY movie_release_year ORDER BY COUNT(movie_id) DESC LIMIT 1"
    },
    {
        "question": "Who is the director that made the most movies? Give the director's id.",
        "evidence": "director that made the most movies refers to MAX(COUNT(movie_id))",
        "SQL": "SELECT director_id FROM movies GROUP BY director_id ORDER BY COUNT(movie_id) DESC LIMIT 1"
    },
    {
        "question": "How many movies did the director of the highest movie popularity make?",
        "evidence": "highest movie popularity refers to MAX(movie_popularity)",
        "SQL": "SELECT COUNT(movie_id) FROM movies WHERE director_id = ( SELECT director_id FROM movies ORDER BY movie_popularity DESC LIMIT 1 )"
    },
    {
        "question": "What's the number of the paying subscribers when rating a movie after the year 2014?",
        "evidence": "paying subscribers refers to user_has_payment_method = 1; rating a movie after the year 2014 refers to rating_date_utc>'2014%'",
        "SQL": "SELECT COUNT(user_subscriber) FROM ratings_users WHERE user_has_payment_method = 1 AND rating_date_utc > '2014%'"
    },
    {
        "question": "Who was the earliest user created a list but didn't get any followers? Give the user ID.",
        "evidence": "earliest user created a list refers to MIN(list_creation_date_utc); didn't get any followers refers to user_subscriber = 0",
        "SQL": "SELECT user_id FROM lists_users WHERE user_subscriber = 0 ORDER BY list_creation_date_utc LIMIT 1"
    },
    {
        "question": "Give the number of followers for the user who posted the most lists.",
        "evidence": "number of followers refers to user_subscriber; posted the most lists refers to MAX(COUNT(list_id))",
        "SQL": "SELECT SUM(T1.list_followers) FROM lists AS T1 INNER JOIN lists_users AS T2 ON T1.list_id = T2.list_id GROUP BY T1.user_id ORDER BY COUNT(T1.list_id) DESC LIMIT 1"
    },
    {
        "question": "How many followers did the user who posted the list \"Non-American Films about World War II\" have?",
        "evidence": "the list \"Non-American Films about World War II\" refers to list_title = 'Non-American Films about World War II'",
        "SQL": "SELECT SUM(T2.list_followers) FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_title LIKE 'Non-American Films about World War II'"
    },
    {
        "question": "What's the number of users gave the movie \"Downfall\" a rating of \"4\"?",
        "evidence": "movie \"Downfall\" refers to movie_title = 'Downfall'; rating of \"4\" refers to rating_score = 4",
        "SQL": "SELECT COUNT(T1.user_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'Downfall' AND T1.rating_score = 4"
    },
    {
        "question": "Give the name of the movie that got the most \"5\" ratings.",
        "evidence": "5 ratings refers to rating_score = 5; name of the movie refers to movie_title",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_score = 5"
    },
    {
        "question": "Which movie got the most critic comments? Give the name of the movie.",
        "evidence": "name of the movie refers to movie_title; most critic comments refers to MAX(critic_comments)",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id GROUP BY T2.movie_title ORDER BY COUNT(T1.critic_comments) DESC LIMIT 1"
    },
    {
        "question": "Show the avatar of the user who gave the rating at 2019/10/17 1:36:36.",
        "evidence": "at 2019/10/17 1:36:36 refers to rating_timestamp_utc = '2019/10/17 1:36:36'; avatar of the user refers to user_avatar_image_url\n\n",
        "SQL": "SELECT T2.user_avatar_image_url FROM ratings AS T1 INNER JOIN lists_users AS T2 ON T1.user_id = T2.user_id WHERE T1.rating_timestamp_utc LIKE '2019-10-17 01:36:36'"
    },
    {
        "question": "Show the portrait picture of the user who created the list \"Vladimir Vladimirovich Nabokov\".",
        "evidence": "the list \"Vladimir Vladimirovich Nabokov\" refers to list_title = 'Vladimir Vladimirovich Nabokov'; portrait picture refers to user_avatar_image_url",
        "SQL": "SELECT T1.user_avatar_image_url FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_title LIKE 'Vladimir Vladimirovich Nabokov'"
    },
    {
        "question": "For the user who post the list that contained the most number of the movies, is he/she a paying subscriber when creating that list?",
        "evidence": "the list that contained the most number of the movies refers to MAX(list_movie_number); user_has_payment_method = 1 means the user was a paying subscriber when he created the list ; \nuser_has_payment_method = 0 means the user was not a paying subscriber when he created the list \n\n",
        "SQL": "SELECT T1.user_has_payment_method FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_movie_number = ( SELECT MAX(list_movie_number) FROM lists )"
    },
    {
        "question": "Show the head portrait of the user who gave the most \"5\" ratings.",
        "evidence": "head portrait refers to user_avatar_image_url; \"5\" ratings refers to rating_score = 5",
        "SQL": "SELECT T2.user_avatar_image_url FROM ratings AS T1 INNER JOIN lists_users AS T2 ON T1.user_id = T2.user_id WHERE T1.rating_score = 5"
    },
    {
        "question": "How many critics were given to the movie that got the most movie popularity number.",
        "evidence": "most movie popularity number refers to MAX(movie_popularity)",
        "SQL": "SELECT COUNT(T1.critic) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_popularity = ( SELECT MAX(movie_popularity) FROM movies )"
    },
    {
        "question": "Who gave a \"4\" rating to the movie \"Freaks\" at 2013/5/4 6:33:32? Give his/her user id.",
        "evidence": "4 rating refers to rating_score = 4; the movie \"Freaks\" refers to movie_title = 'Freaks' ; at 2013/5/4 6:33:32 refers to rating_timestamp_utc = '2013-05-04 06:33:32'",
        "SQL": "SELECT T1.user_id FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE rating_score = 4 AND rating_timestamp_utc LIKE '2013-05-04 06:33:32' AND T2.movie_title LIKE 'Freaks'"
    },
    {
        "question": "Give the url of movie which was rated 5 on 2013/5/3 5:11:17.",
        "evidence": "rated 5 refers to rating_score = 5; on 2013/5/3 5:11:17 refers to rating_timestamp_utc = '2013-05-03 05:11:17'",
        "SQL": "SELECT T2.movie_url FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE rating_score = 5 AND rating_timestamp_utc LIKE '2013-05-03 05:11:17'"
    },
    {
        "question": "For the 1998 movie which got the highest popularity, how many \"4\" rating did the movie get?",
        "evidence": "1998 movie refers to movie_release_year = '1998'; the highest popularity refers to MAX(movie_popularity) ; \"4\" rating refers to rating_score = 4",
        "SQL": "SELECT COUNT(T2.movie_title) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_score = 4 AND T2.movie_release_year = 1998 ORDER BY T2.movie_popularity DESC LIMIT 1"
    },
    {
        "question": "From all the movies that got more than 13000 popularity number, which one had the least ratings.",
        "evidence": "more than 13000 popularity number refers to movie_popularity > 13000; least ratings refers to MIN(rating_score)",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_popularity > 13000 ORDER BY T1.rating_score LIMIT 1"
    },
    {
        "question": "How many paying subscribers gave a rating to the movie \"One Flew Over the Cuckoo's Nest\"?",
        "evidence": "paying subscribers refer to user_has_payment_method = 1; movie \"One Flew Over the Cuckoo's Nest\" refers to movie_id = 'One Flew Over the Cuckoo''s Nest'",
        "SQL": "SELECT COUNT(T1.user_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id INNER JOIN ratings_users AS T3 ON T1.user_id = T3.user_id WHERE T2.movie_title = 'One Flew Over the Cuckoo''s Nest' AND T3.user_has_payment_method = 1"
    },
    {
        "question": "For the lists that got more than 3000 followers, how many did the users who created those lists are paying subscribers?",
        "evidence": "got more than 3000 followers refers to list_followers > 3000; paying subscribers refer to user_has_payment_method = 1",
        "SQL": "SELECT COUNT(T1.user_id) FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_followers > 3000 AND T1.user_has_payment_method = 1"
    },
    {
        "question": "Which 1988 movie got the most ratings?",
        "evidence": "1988 movie refers to movie_release_year = '1998'; most ratings refers to MAX(rating_score)",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_release_year = 1988 ORDER BY T1.rating_score DESC LIMIT 1"
    },
    {
        "question": "For all the movies that were released in 1995, how many lower than 3 ratings did the most popularity movie had?",
        "evidence": "released in 1995 refers to movie_release_year = '1995'; lower than 3 ratings refers to rating_score <3; most popularity movie refers to MAX(movie_popularity)",
        "SQL": "SELECT COUNT(T1.rating_score) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.rating_score < 3 AND T2.movie_release_year = 1995 AND T2.movie_popularity = ( SELECT MAX(movie_popularity) FROM movies WHERE movie_release_year = 1995 )"
    },
    {
        "question": "What is the percentage of users gave \"5\" to the movie \"Go Go Tales\"?",
        "evidence": "movie \"Go Go Tales\" refers to movie_title = 'Go Go Tales'; gave \"5\" refers to rating_score = 5; percentage refers to DIVIDE(COUNT(rating_score = 5),COUNT(rating_score))*100",
        "SQL": "SELECT CAST(SUM(CASE WHEN T1.rating_score = 5 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.user_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'Go Go Tales'"
    },
    {
        "question": "Give the percentage of subscribers who rated who rated the movie \"G.I. Jane\".",
        "evidence": "movie \"G.I. Jane\" refers to movie_title = 'G.I. Jane'; subscribers refers to user_subscriber = 1; percentage refers to DIVIDE(COUNT(user_subscriber = 1),COUNT(user_subscriber))*100",
        "SQL": "SELECT CAST(SUM(CASE WHEN T3.user_subscriber = 1 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id INNER JOIN lists_users AS T3 ON T1.user_id = T3.user_id WHERE T2.movie_title = 'G.I. Jane'"
    },
    {
        "question": "For all the users who gave \"A Shot in the Dark\" a rating, how many percent of them is a paying subscriber?",
        "evidence": "\"A Shot in the Dark\" refers to movie_title = 'A Shot in the Dark'; paying subscriber refers to user_has_payment_method = 1; percentage refers to DIVIDE(COUNT(user_has_payment_method = 1),COUNT(user_has_payment_method))*100",
        "SQL": "SELECT CAST(SUM(CASE WHEN T1.user_has_payment_method = 1 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id INNER JOIN lists_users AS T3 ON T1.user_id = T3.user_id WHERE T2.movie_title = 'A Shot in the Dark'"
    },
    {
        "question": "Name all the list titles created by user 4208563.",
        "evidence": "user 4208563 refers to user_id = 4208563",
        "SQL": "SELECT list_title FROM lists WHERE user_id LIKE 4208563"
    },
    {
        "question": "Among the lists created in 2016, which is the list that was updated most recently.",
        "evidence": "created in 2016 refers to list_creation_timestamp_utc like '2016%'; updated most recently refers to MAX(list_update_timestamp_utc)",
        "SQL": "SELECT list_title FROM lists WHERE strftime('%Y', list_update_timestamp_utc) = '2016' ORDER BY list_update_timestamp_utc DESC LIMIT 1"
    },
    {
        "question": "What is the percentage of list created by user who was a subscriber when he created the list?",
        "evidence": "was a subscriber refers to user_subscriber = 1; percentage refers to DIVIDE(COUNT(user_subscriber = 1),COUNT(list_id))",
        "SQL": "SELECT CAST(SUM(CASE WHEN user_subscriber = 1 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(list_id) FROM lists_users"
    },
    {
        "question": "Name all lists created by a user who was a subcriber when created the list.",
        "evidence": "was a subscriber refers to user_subscriber = 1",
        "SQL": "SELECT DISTINCT T2.list_id FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T1.user_subscriber = 1"
    },
    {
        "question": "Provide list titles created by user who are eligible for trial when he created the list.",
        "evidence": "eligible for trial refers to user_eligible_for_trial = 1",
        "SQL": "SELECT DISTINCT T2.list_title FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T1.user_eligible_for_trial = 1"
    },
    {
        "question": "Among the lists with at least one follower, how many were created by user who was subscriber when created the list?",
        "evidence": "lists with at least one follower refers to list_followers > = 1; was a subscriber refers to user_subscriber = 1",
        "SQL": "SELECT COUNT(T1.list_id) FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_followers >= 1 AND T1.user_subscriber = 1"
    },
    {
        "question": "For all list titles with at least 200 movies in the list, what is their average number of followers?",
        "evidence": "at least 200 movies in the list refers to list_movie_number > 200; average number of followers refers to avg(list_followers)",
        "SQL": "SELECT AVG(list_followers) FROM lists WHERE list_movie_number > 200"
    },
    {
        "question": "List all the titles created by user who was a subsriber when he created the list and have less than 50 movies in the list.",
        "evidence": "have less than 50 movies in the list refers to list_movie_number <50; was a subscriber refers to user_subscriber = 1",
        "SQL": "SELECT DISTINCT T2.list_title FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_movie_number < 50 AND T1.user_subscriber = 1"
    },
    {
        "question": "Which title list has not been updated for the longest period of time? State how long it has not been updated?",
        "evidence": "not been updated for the longest period of time refers to MIN(list_update_timestamp_utc); how long it has not been updated refers to SUBTRACT(CURRENT_TIMESTAMP, list_update_timestamp_utc)",
        "SQL": "SELECT list_title , datetime(CURRENT_TIMESTAMP, 'localtime') - datetime(list_update_timestamp_utc) FROM lists ORDER BY list_update_timestamp_utc LIMIT 1"
    },
    {
        "question": "Who is the user who created the list titled 'Sound and Vision'? Was he a subcriber when he created the list?",
        "evidence": "list titled 'Sound and Vision' refers to list_title = 'Sound and Vision'; user_subscriber = 1 means the user was a subscriber when he rated the movie; user_subscriber = 0 means the user was not a subscriber when he rated the movie\n\n\n\n",
        "SQL": "SELECT T1.user_id, T1.user_subscriber FROM lists_users AS T1 INNER JOIN lists AS T2 ON T1.list_id = T2.list_id WHERE T2.list_title LIKE 'Sound and Vision'"
    },
    {
        "question": "For the list with more than 200 followers, state the title and how long the list has been created?",
        "evidence": "more than 200 followers refers to list_followers >200; how long the list has been created refers to SUBTRACT(CURRENT_TIMESTAMP,list_creation_timestamp_utc)",
        "SQL": "SELECT list_title , 365 * (strftime('%Y', 'now') - strftime('%Y', list_creation_timestamp_utc)) + 30 * (strftime('%m', 'now') - strftime('%m', list_creation_timestamp_utc)) + strftime('%d', 'now') - strftime('%d', list_creation_timestamp_utc) FROM lists WHERE list_followers > 200"
    },
    {
        "question": "Among all movies in the list, calculate the percentage of movies that were never been rated?",
        "evidence": "percentage of movies that were never been rated refers to DIVIDE(COUNT(main_movies.movie_id ! = main_ratings.movie_id),COUNT(movie_id))",
        "SQL": "SELECT CAST(SUM(CASE WHEN T2.movie_id IS NULL THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T2.movie_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id"
    },
    {
        "question": "List all movies rated by user 39115684. State the title, rating date and rating score.",
        "evidence": "user 39115684 refers to user_id = 39115684; title refers to movie_title; rating date refers to rating_timestamp_utc\n",
        "SQL": "SELECT T2.movie_title, T1.rating_timestamp_utc, T1.rating_score FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.user_id = 39115684"
    },
    {
        "question": "Between 1970 to 1980, how many movies with a popularity of more than 11,000 were released?",
        "evidence": "Between 1970 to 1980 refers to movie_release_year between 1970 and 1980; popularity of more than 11,000 refers movie_popularity >11000",
        "SQL": "SELECT COUNT(movie_id) FROM movies WHERE movie_release_year BETWEEN '1970' AND '1980' AND movie_popularity > 11000"
    },
    {
        "question": "How many movies directed by Felipe Cazals was realeased on 1976?",
        "evidence": "directed by Felipe Cazals refers to director_name = 'Felipe Cazals' ; realeased on 1976 refers to movie_release_year = 1976",
        "SQL": "SELECT COUNT(movie_id) FROM movies WHERE movie_release_year = 1976 AND director_name LIKE 'Felipe Cazals'"
    },
    {
        "question": "What is the URL to the movie director page on Mubi of the movie titled \"Red Blooded American Girl\"",
        "evidence": "movie titled \"Red Blooded American Girl\" refers to movie_title = 'Red Blooded American Girl'",
        "SQL": "SELECT director_url FROM movies WHERE movie_title LIKE 'Red Blooded American Girl'"
    },
    {
        "question": "What is the name of the list that was updated most recently?",
        "evidence": "updated most recently refers to MAX(list_update_date_utc)",
        "SQL": "SELECT list_title FROM lists WHERE list_update_timestamp_utc = ( SELECT list_update_timestamp_utc FROM lists ORDER BY list_update_timestamp_utc DESC LIMIT 1 )"
    },
    {
        "question": "Who created the list that has 142 comments? Indicate the user id of the user, if there are multiple lists with 142 comments, list the user id of the person who created the list",
        "evidence": "list that has 142 comments refers to list_comments = 142",
        "SQL": "SELECT user_id FROM lists WHERE list_comments = 142"
    },
    {
        "question": "What is Jeannot Szwarc's most popular movie and what is its average rating score?",
        "evidence": "Jeannot Szwarc's refers to director_name = 'Jeannot Szwarc'; most popular movie refers to MAX(movie_popularity); average rating score refers to avg(rating_score)",
        "SQL": "SELECT T2.movie_title, AVG(T1.rating_score) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.director_name = 'Jeannot Szwarc' ORDER BY T2.movie_popularity DESC LIMIT 1"
    },
    {
        "question": "Who is the director that directed the highest number of movies in the 70s? If there are multiple directors with the same amount of movies, list all of their names and indicate the highest rating score that those movies got from the users.",
        "evidence": "highest number of movies COUNT(T1.movie_id); in the 70s refers to movie_release_year between 1970 and 1979",
        "SQL": "SELECT T2.director_name, T1.rating_score FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_release_year BETWEEN 1970 AND 1979 GROUP BY T2.director_id ORDER BY COUNT(T2.movie_id) DESC LIMIT 1"
    },
    {
        "question": "Between 1/1/2010 to 12/31/2020, how many users, who were a trialist when they created the list, gave the movie \"The Secret Life of Words\" a rating score of 3?",
        "evidence": "Between 1/1/2010 to 12/31/2020 refers to rating_timestamp_utc between '2010-01-01%' and '2020-12-31%'; a trialist refers to user_trialist = 1; movie \"The Secret Life of Words\" refers to movie_title = 'The Secret Life of Words'; rating score of 3 refers to rating_score = 3",
        "SQL": "SELECT COUNT(T1.user_id) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T2.movie_title = 'The Secret Life of Words' AND T1.rating_score = 3 AND T1.user_trialist = 0 AND T1.rating_timestamp_utc BETWEEN '2010%' AND '2020%'"
    },
    {
        "question": "What is the name of the movie whose critic received the highest amount of likes? Indicate the URL to the rating on Mubi.",
        "evidence": "critic received the highest amount of likes refers to MAX(critic_likes);",
        "SQL": "SELECT T2.movie_title, T1.rating_url FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id ORDER BY T1.critic_likes DESC LIMIT 1"
    },
    {
        "question": "What are the top 5 most popular movies of the 21st century? Indicate how many users gave it a rating score of 5.",
        "evidence": "most popular movies refers to MAX(movie_popularity); rating score of 5 refers to rating_score = 5; movies of the 21st century refers to movie_release_year> = 2000",
        "SQL": "SELECT DISTINCT T2.movie_id, SUM(T1.rating_score = 5) FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id ORDER BY T2.movie_popularity DESC LIMIT 5"
    },
    {
        "question": "What is the average number of followers of the lists created by the user who rated the movie \"Pavee Lackeen: The Traveller Girl\" on 3/27/2011 at 2:06:34 AM?",
        "evidence": "average number of followers refers to AVG(list_followers); movie \"Pavee Lackeen: The Traveller Girl\" refers to movie_title = 'Pavee Lackeen: The Traveller Girl'; on 3/27/2011 at 2:06:34 AM refers to rating_timestamp_utc = '2011-03-27 02:06:34'",
        "SQL": "SELECT CAST(SUM(T4.list_followers) AS REAL) / COUNT(T2.list_id) FROM ratings AS T1 INNER JOIN lists_users AS T2 ON T1.user_id = T2.user_id INNER JOIN movies AS T3 ON T1.movie_id = T3.movie_id INNER JOIN lists AS T4 ON T2.list_id = T4.list_id WHERE T3.movie_title LIKE 'Pavee Lackeen: The Traveller Girl' AND T1.rating_timestamp_utc LIKE '2011-03-27 02:06:34'"
    },
    {
        "question": "Between 1/1/2017 to 12/31/2017, how many users who were eligible for trial when they rated the movie \"Patti Smith: Dream of Life\"and what is the image URL to the movie on Mubi?",
        "evidence": "Between 1/1/2017 to 12/31/2017 refers to rating_timestamp_utc between '2017-01-01 00:00:00' and '2017-12-31 00:00:00'; eligible for trial refers to user_eligible_for_trial = 1; movie \"Patti Smith: Dream of Life\" refers to movie_title = 'Patti Smith: Dream of Life'",
        "SQL": "SELECT COUNT(T1.user_id), T2.movie_image_url FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE datetime(T1.rating_timestamp_utc) BETWEEN '2017-01-01 00:00:00' AND '2017-12-31 00:00:00'"
    },
    {
        "question": "What is the average number of number of movies added to the lists of user 8516503? Indicate how many movies did he/she give a rating score of 5.",
        "evidence": "average number of number of movies refers to AVG(list_movie_number); user 8516503 refers to user_id = 8516503; rating score of 5 refers to rating_score = 5",
        "SQL": "SELECT AVG(T3.list_movie_number) , SUM(CASE WHEN T1.rating_score = 5 THEN 1 ELSE 0 END) FROM ratings AS T1 INNER JOIN lists_users AS T2 ON T1.user_id = T2.user_id INNER JOIN lists AS T3 ON T2.user_id = T3.user_id WHERE T1.user_id = 8516503"
    },
    {
        "question": "Who is the director of the most popular movie of all time and when was it released? Indicate the average rating score of the users who were on a trialist when they rated the movie.",
        "evidence": "most popular movie of all time refers to MAX(movie_popularity); a trialist refers to user_trialist = 1; average rating score = AVG(rating_score)",
        "SQL": "SELECT T1.director_name, T1.movie_release_year , SUM(T2.rating_score) / COUNT(T2.user_id) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id WHERE T2.user_trialist = 1 ORDER BY T1.movie_popularity DESC LIMIT 1"
    },
    {
        "question": "What is the name of the movie that was rated recently by user 57756708?",
        "evidence": "user 57756708 refers to user_id = 57756708; rated recently refers to MAX(rating_timestamp_utc)",
        "SQL": "SELECT T2.movie_title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id WHERE T1.user_id = 57756708 ORDER BY T1.rating_timestamp_utc DESC LIMIT 1"
    },
    {
        "question": "What are the top 10 oldest movies and what are the average rating score for each movie? Indicate the name of the director and when the movies were released.",
        "evidence": "the average rating score refers to AVG(T2.rating_score); oldest movies refers to MIN(rating_timestamp_utc)",
        "SQL": "SELECT T2.movie_id, AVG(T1.rating_score), T2.director_name, T2.movie_release_year FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T2.movie_id ORDER BY T1.rating_timestamp_utc ASC LIMIT 10"
    }
}

2. DB Schema of Samples
{
    CREATE TABLE lists (
        user_id                     INTEGER REFERENCES lists_users (user_id),
        list_id                     INTEGER NOT NULL
                                            PRIMARY KEY,
        list_title                  TEXT,
        list_movie_number           INTEGER,
        list_update_timestamp_utc   TEXT,
        list_creation_timestamp_utc TEXT,
        list_followers              INTEGER,
        list_url                    TEXT,
        list_comments               INTEGER,
        list_description            TEXT,
        list_cover_image_url        TEXT,
        list_first_image_url        TEXT,
        list_second_image_url       TEXT,
        list_third_image_url        TEXT
    );

    CREATE TABLE lists_users (
        user_id                 INTEGER NOT NULL,
        list_id                 INTEGER NOT NULL,
        list_update_date_utc    TEXT,
        list_creation_date_utc  TEXT,
        user_trialist           INTEGER,
        user_subscriber         INTEGER,
        user_avatar_image_url   TEXT,
        user_cover_image_url    TEXT,
        user_eligible_for_trial TEXT,
        user_has_payment_method TEXT,
        PRIMARY KEY (
            user_id,
            list_id
        ),
        FOREIGN KEY (
            list_id
        )
        REFERENCES lists (list_id),
        FOREIGN KEY (
            user_id
        )
        REFERENCES lists (user_id) 
    );

    CREATE TABLE movies (
        movie_id             INTEGER NOT NULL
                                     PRIMARY KEY,
        movie_title          TEXT,
        movie_release_year   INTEGER,
        movie_url            TEXT,
        movie_title_language TEXT,
        movie_popularity     INTEGER,
        movie_image_url      TEXT,
        director_id          TEXT,
        director_name        TEXT,
        director_url         TEXT
    );

    CREATE TABLE ratings (
        movie_id                INTEGER,
        rating_id               INTEGER,
        rating_url              TEXT,
        rating_score            INTEGER,
        rating_timestamp_utc    TEXT,
        critic                  TEXT,
        critic_likes            INTEGER,
        critic_comments         INTEGER,
        user_id                 INTEGER,
        user_trialist           INTEGER,
        user_subscriber         INTEGER,
        user_eligible_for_trial INTEGER,
        user_has_payment_method INTEGER,
        FOREIGN KEY (
            movie_id
        )
        REFERENCES movies (movie_id),
        FOREIGN KEY (
            user_id
        )
        REFERENCES lists_users (user_id),
        FOREIGN KEY (
            rating_id
        )
        REFERENCES ratings (rating_id),
        FOREIGN KEY (
            user_id
        )
        REFERENCES ratings_users (user_id) 
    );

    CREATE TABLE ratings_users (
        user_id                 INTEGER REFERENCES lists_users (user_id),
        rating_date_utc         TEXT,
        user_trialist           INTEGER,
        user_subscriber         INTEGER,
        user_avatar_image_url   TEXT,
        user_cover_image_url    TEXT,
        user_eligible_for_trial INTEGER,
        user_has_payment_method INTEGER
    );
}

3. Schema Description
{
	list
	{
	original_column_name,column_name,column_description,data_format,value_description
	user_id,,ID related to the user who created the list.,integer,
	list_id,,ID of the list on Mubi,integer,
	list_title,,Name of the list,text,
	list_movie_number,,Number of movies added to the list,integer,
	list_update_timestamp_utc,,Last update timestamp for the list,text,
	list_creation_timestamp_utc,,Creation timestamp for the list,text,
	list_followers,,Number of followers on the list,integer,
	list_url,,URL to the list page on Mubi,text,
	list_comments,,Number of comments on the list,integer,
	list_description,,List description made by the user,text,
	list_cover_image_url,,,,
	list_first_image_url,,,,
	list_second_image_url,,,,
	list_third_image_url,,,,
	},
	lists_users
	{
	original_column_name,column_name,column_description,data_format,value_description
	user_id,,ID related to the user who created the list.,integer,
	list_id,,ID of the list on Mubi,integer,
	list_update_date_utc,,Last update date for the list,text,YYYY-MM-DD
	list_creation_date_utc,,Creation date for the list,text,YYYY-MM-DD
	user_trialist,,whether the user was a tralist when he created the list ,integer,"1 = the user was a trialist when he created the list
	 0 = the user was not a trialist when he created the list"
	user_subscriber,,whether the user was a subscriber when he created the list ,integer,"1 = the user was a subscriber when he created the list 
	0 = the user was not a subscriber when he created the list"
	user_avatar_image_url,,User profile image URL on Mubi,text,
	user_cover_image_url,,User profile cover image URL on Mubi,text,
	user_eligible_for_trial,,whether the user was eligible for trial when he created the list ,text,"1 = the user was eligible for trial when he created the list 
	0 = the user was not eligible for trial when he created the list"
	user_has_payment_method ,,whether the user was a paying subscriber when he created the list ,text,"1 = the user was a paying subscriber when he created the list 
	0 = the user was not a paying subscriber when he created the list "
	},
	movies
	{
	original_column_name,column_name,column_description,data_format,value_description
	movie_id,,ID related to the movie on Mubi,integer,
	movie_title,,Name of the movie,text,
	movie_release_year,,Release year of the movie,integer,
	movie_url,,URL to the movie page on Mubi,text,
	movie_title_language,,"By default, the title is in English.",text,Only contains one value which is 'en'
	movie_popularity,,Number of Mubi users who love this movie,integer,
	movie_image_url,,Image URL to the movie on Mubi,text,
	director_id,,ID related to the movie director on Mubi,text,
	director_name,,Full Name of the movie director,text,
	director_url ,,URL to the movie director page on Mubi,text,
	},
	ratings
	{
	original_column_name,column_name,column_description,data_format,value_description
	movie_id,,Movie ID related to the rating,integer,
	rating_id,,Rating ID on Mubi,integer,
	rating_url,,URL to the rating on Mubi,text,
	rating_score,,Rating score ranging from 1 (lowest) to 5 (highest),integer,"commonsense evidence:
	The score is proportional to the user's liking.
	The higher the score is, the more the user likes the movie"
	rating_timestamp_utc ,,Timestamp for the movie rating made by the user on Mubi,text,
	critic,,Critic made by the user rating the movie. ,text,"If value = ""None"", the user did not write a critic when rating the movie."
	critic_likes,,Number of likes related to the critic made by the user rating the movie,integer,
	critic_comments,,Number of comments related to the critic made by the user rating the movie,integer,
	user_id,,ID related to the user rating the movie,integer,
	user_trialist ,,whether user was a tralist when he rated the movie,integer,"1 = the user was a trialist when he rated the movie 
	0 = the user was not a trialist when he rated the movie"
	user_subscriber,,,integer,
	user_eligible_for_trial,,,integer,
	user_has_payment_method,,,integer,
	},
	ratings_users
	{
	original_column_name,column_name,column_description,data_format,value_description
	user_id,,ID related to the user rating the movie,integer,
	rating_date_utc,,Rating date for the movie rating.,text,YYYY-MM-DD
	user_trialist,,whether the user was a trialist when he rated the movie,integer,"1 = the user was a trialist when he rated the movie
	 0 = the user was not a trialist when he rated the movie"
	user_subscriber,,whether the user was a subscriber when he rated the movie,integer,"1 = the user was a subscriber when he rated the movie 
	0 = the user was not a subscriber when he rated the movie"
	user_avatar_image_url,,URL to the user profile image on Mubi,text,
	user_cover_image_url,,URL to the user profile cover image on Mubi,text,
	user_eligible_for_trial,,whether the user was eligible for trial when he rated the movie,integer,"1 = the user was eligible for trial when he rated the movie
	 0 = the user was not eligible for trial when he rated the movie"
	user_has_payment_method ,,whether the user was a paying subscriber when he rated the movie,integer,"1 = the user was a paying subscriber when he rated the movie 
	0 = the user was not a paying subscriber when he rated"
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
