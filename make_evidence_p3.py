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
    prompt = """Purpose: Create an evidence to aid text-to-SQL tasks
action
  1. Please refer to the given question, evidence, and SQL pair and the DB Schema of samples, and schema description.
  2. For the given question, schema and schema description, generate evidence in one sentence to help text-to-sql.
  3. Skip the description and just print out evidence.

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
