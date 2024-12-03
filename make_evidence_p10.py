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

def make_prompt(question, concat_schema):
    prompt = """### You are an assistant to a data scientist. 
You can capture the link between the question and corresponding database and perfectly generate valid evidence to assist answer the question. 
Your objective is to generate evidence by analyzing and understanding the essence of the given question, database schema, database column descriptions. 
This evidence step is essential for extracting the correct information from the database and finding the answer for the question.

### Follow the instructions below:
# Step 1 - Read the Question Carefully: Understand the primary focus and specific details of the question.
# Step 2 - Analyze the Database Schema: Examine the database schema, database column descriptions and sample descriptions. Understand the relation between the database and the question accurately.
# Step 3 - Generate evidence: Write evidence corresponding to the given question by combining the sense of question, and database items.

### samples ####################################################
1. DB Schema of Samples
{
    CREATE TABLE playstore (
        App              TEXT,    ### original_column_name: App, column_name: , column_description: Application name, data_format: text, value_description: 
        Category         TEXT,    ### original_column_name: Category, column_name: , column_description: Category the app belongs to, data_format: text, value_description: FAMILY 18% GAME 11% Other (7725) 71%
        Rating           REAL,    ### original_column_name: Rating, column_name: , column_description: Overall user rating of the app (as when scraped), data_format: real, value_description: 
        Reviews          INTEGER,    ### original_column_name: Reviews, column_name: , column_description: Number of user reviews for the app (as when scraped), data_format: integer, value_description: 
        Size             TEXT,    ### original_column_name: Size, column_name: , column_description: Size of the app (as when scraped), data_format: text, value_description: Varies with device 16% 11M 2% Other (8948) 83%
        Installs         TEXT,    ### original_column_name: Installs, column_name: , column_description: Number of user downloads/installs for the app (as when scraped), data_format: text, value_description: 1,000,000+ 15% 10,000,000+ 12% Other (8010) 74%
        Type             TEXT,    ### original_column_name: Type, column_name: , column_description: Paid or Free, data_format: text, value_description: Only has 'Paid' and 'Free' Free 93% Paid 7%
        Price            TEXT,    ### original_column_name: Price, column_name: , column_description: Price of the app (as when scraped), data_format: text, value_description: 0 93% $0.99 1% Other (653) 6% commonsense evidence: Free means the price is 0.
        [Content Rating] TEXT,    ### original_column_name: Content Rating, column_name: , column_description: Age group the app is targeted at - Children / Mature 21+ / Adult, data_format: text, value_description: Everyone 80% Teen 11% Other (919) 8% commonsense evidence: According to Wikipedia, the different age groups are defined as: Children: age<12~13 Teen: 13~19 Adult/Mature: 21+ 
        Genres           TEXT    ### original_column_name: Genres, column_name: , column_description: An app can belong to multiple genres (apart from its main category)., data_format: text, value_description: Tools 8% Entertainment 6% Other (9376) 86%
    );

    CREATE TABLE user_reviews (
        App                    TEXT REFERENCES playstore (App),    ### original_column_name: App, column_name: , column_description: Name of app, data_format: text, value_description: 
        Translated_Review      TEXT,    ### original_column_name: Translated_Review, column_name: , column_description: User review (Preprocessed and translated to English), data_format: text, value_description: nan 42% Good 0% Other (37185) 58%
        Sentiment              TEXT,    ### original_column_name: Sentiment, column_name: , column_description: Overall user rating of the app (as when scraped), data_format: text, value_description: commonsense evidence: Positive: user holds positive attitude towards this app Negative: user holds positive attitude / critizes on this app Neural: user holds neural attitude nan: user doesn't comment on this app.
        Sentiment_Polarity     TEXT,    ### original_column_name: Sentiment_Polarity, column_name: Sentiment Polarity, column_description: Sentiment polarity score, data_format: text, value_description: commonsense evidence: ??score >= 0.5 it means pretty positive or pretty like this app. ??0 <= score < 0.5: it means user mildly likes this app. ??-0.5 <= score < 0: it means this user mildly dislikes this app or slightly negative attitude ??score <-0.5: it means this user dislikes this app pretty much.
        Sentiment_Subjectivity TEXT    ### original_column_name: Sentiment_Subjectivity, column_name: Sentiment Subjectivity, column_description: Sentiment subjectivity score, data_format: text, value_description: commonsense evidence: more subjectivity refers to less objectivity, vice versa.
    );
}

2. Question, evidence, and SQL pair samples
{
    {
        "question": "How many apps were last updated in January of 2018? Please write one translated review with positive sentiment for each app, if there's any.",
        "evidence": "updated in January of 2018 refers to Last Updated BETWEEN 'January 1, 2018' and 'January 31, 2018';",
    },
    {
        "question": "How many users mildly likes the 7 Minute Workout app and when was it last updated?",
        "evidence": "mildly likes the app refers to Sentiment_Polarity> = 0 and Sentiment_Polarity<0.5;",
    },
    {
        "question": "How many users holds neutral attitude towards the HTC Weather app? Indicate the app's rating on the Google Play Store.",
        "evidence": "user holds neutral attitude refers to Sentiment = 'Neutral';",
    },
    {
        "question": "What is the name and category of the app with the highest amount of -1 sentiment polarity score?",
        "evidence": "highest amount of -1 sentiment polarity score refers to MAX(Count(Sentiment_Polarity = 1.0))",
    },
    {
        "question": "What is the average sentiment polarity score of the Cooking Fever app? Indicate the age group that the app is targeted at.",
        "evidence": "average sentiment polarity score = AVG(Sentiment_Polarity); age group the app is target at refers to Content Rating;",
    },
    {
        "question": "What is the lowest sentiment polarity score of the Basketball Stars app for people who dislikes the app pretty much and how many downloads does it have?",
        "evidence": "user dislike the app pretty much refers to Sentiment_Polarity<-0.5; number of downloads it has refers to installs;",
    },
    {
        "question": "For the Akinator app, how many reviews have sentiment subjectivity of no more than 0.5 and what is its current version?",
        "evidence": "Sentiment_Subjectivity<0.5; current version refers to Current Ver;",
    },
    {
        "question": "What are the top 5 installed free apps?",
        "evidence": "free app refers to price = 0;",
    },
    {
        "question": "How many of the users hold neutral attitude on \"10 Best Foods for You\" app and what category is this app?",
        "evidence": "neutral attitude refers to Sentiment = 'Neutral';",
    },
    {
        "question": "What are the apps that users pretty like this app and how many installs amount of these apps?",
        "evidence": "users pretty much likes the app refers to Sentiment_Polarity = 'Positive';",
    },
    {
        "question": "How many apps that are only compatible with Android ver 8.0 and above? List down the users' sentiment of these apps.",
        "evidence": "compatible with android ver 8.0 refers \"Android Ver\" = '8.0 and up';",
    },
    {
        "question": "Which apps have not been updated since year 2015 and what kind of sentiment users hold on it?",
        "evidence": "not been updated since year 2015 refers to \"Last Updated\"<'January 1, 2015';",
    },
    {
        "question": "How many of the reviews for the app \"Brit + Co\" have a comment?",
        "evidence": "Brit + Co refers to App = 'Brit + Co'; have a comment refers to Translated Review NOT null;",
    },
    {
        "question": "List the top 5 shopping apps with the most reviews.",
        "evidence": "shopping apps refers to Genre = 'Shopping';",
    },
    {
        "question": "How many neutral reviews does the app \"Dino War: Rise of Beasts\" have?",
        "evidence": "neutral reviews refers to Sentiment = 'Neutral';",
    },
    {
        "question": "List all the negative comments on the \"Dog Run - Pet Dog Simulator\" app.",
        "evidence": "negative comment refers to Sentiment = 'Negative';",
    },
    {
        "question": "How many negative comments are there in all the apps with 100,000,000+ installs?",
        "evidence": "negative comment refers to Sentiment = 'Negative'; Installs = '100,000,000+';",
    },
    {
        "question": "What are the content ratings for the apps that have \"gr8\" in their comments?",
        "evidence": "app with gr8 in their comments refers to Translated_Review LIKE '%gr8%';",
    },
    {
        "question": "Which Photography app has the highest total Sentiment subjectivity score?",
        "evidence": "Photography app refers to Genre = 'Photography'; highest total sentiment subjectivity score = MAX(sum(Sentiment_Subjectivity));",
    },
    {
        "question": "List all the comments on the lowest rated Mature 17+ app.",
        "evidence": "comments refers to Translated_Review; lowest rated refers to Rating = 1; Mature 17+ refers to Content Rating = 'Mature 17+ ';",
    }
}
##################################################################

### question ####################################################
"""
    prompt += f"""1. schema of question
{{
    {concat_schema}}}
    
2. question
{{
    "question": "{question}",
    "evidence": 
}}

### Please skip the description and just print out evidence.

Let's think step by step and generate evidence.

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
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' order by tbl_name;")
    schemas = cursor.fetchall()
    schema = ""
    for sc in schemas:
        schema += (sc[0] + "\n\n")
    return schema


def read_schema_description(db_path):
    files = sorted(os.listdir(db_path))
    schema_description = ""
    
    for csv_file in files:
        file_path = os.path.join(db_path, csv_file)

        try:
            with open(file_path, 'r', encoding='cp1252') as f:
                lines = [line.replace('\x00', '').replace('\n', ' ').replace('\r', ' ').strip() for line in f if line.strip()]

                csv_reader = csv.reader(lines)
                headers = next(csv_reader)

                for row in csv_reader:
                    if row:
                        row_description = ", ".join([f"{header}: {value}" for header, value in zip(headers, row)])
                        row_description = "   ### " + row_description
                        schema_description += row_description + '\n'

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
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

        schema_lines = schema.splitlines()
        schema_description_lines = schema_description.splitlines()

        concat_schema_lines = []
        schema_description_index = 0

        for schema_line in schema_lines:
            if schema_line \
            and not schema_line.startswith("CREATE") \
            and not schema_line.startswith("--") \
            and not schema_line.startswith(")") \
            and not schema_line.startswith("(") \
            and not schema_line.strip().lower().startswith("unique") \
            and not schema_line.strip().lower().startswith("references") \
            and not schema_line.strip().lower().startswith("on update cascade") \
            and not schema_line.strip().lower().startswith("primary key") \
            and not schema_line.strip().lower().startswith("constraint") \
            and not schema_line.strip().lower().startswith("foreign key") \
            and not schema_line.strip().lower().startswith("unique"):
                concat_schema_lines.append(schema_line + schema_description_lines[schema_description_index])
                schema_description_index += 1
            else:
                concat_schema_lines.append(schema_line)

        concat_schema = "\n".join(concat_schema_lines)

        prompt = make_prompt(data["question"], concat_schema)

        # if i == 0 or temp_db_id != data['db_id']:
        #     temp_db_id = data['db_id']
        #     print(data['db_id'])
        #     print(concat_schema)

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