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

### sample 1 ####################################################
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

2. Question and evidence pair samples
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
        "question": "What is the lowest sentiment polarity score of the Basketball Stars app for people who dislikes the app pretty much and how many downloads does it have?",
        "evidence": "user dislike the app pretty much refers to Sentiment_Polarity<-0.5; number of downloads it has refers to installs;",
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
    }
}
##################################################################


### sample 1 ####################################################
1. DB Schema of Samples
{
    CREATE TABLE authors
    (
        au_id    TEXT    ### original_column_name: au_id, column_name: author id, column_description: unique number identifying authors, data_format: text, value_description: 
            primary key,
        au_lname TEXT not null,    ### original_column_name: au_lname, column_name: author last name, column_description: author last name, data_format: text, value_description: 
        au_fname TEXT not null,    ### original_column_name: au_fname, column_name: author first name, column_description: author first name, data_format: text, value_description: 
        phone    TEXT    not null,    ### original_column_name: phone, column_name: , column_description: phone number, data_format: text, value_description: 
        address  TEXT,    ### original_column_name: address, column_name: , column_description: address, data_format: text, value_description: 
        city     TEXT,    ### original_column_name: city, column_name: , column_description: city , data_format: text, value_description: 
        state    TEXT,    ### original_column_name: state, column_name: , column_description: state , data_format: text, value_description: 
        zip      TEXT,    ### original_column_name: zip, column_name: , column_description: zip code, data_format: text, value_description: 
        contract TEXT     not null    ### original_column_name: contract, column_name: , column_description: contract status, data_format: text, value_description: commonsense evidence: 0: not on the contract 1: on the contract
    )

    CREATE TABLE discounts
    (
        discounttype TEXT   not null,    ### original_column_name: discounttype, column_name: discount type, column_description: discount type, data_format: text, value_description: 
        stor_id      TEXT,    ### original_column_name: stor_id, column_name: store id, column_description: store id, data_format: text, value_description: 
        lowqty       INTEGER,    ### original_column_name: lowqty, column_name: low quantity, column_description: low quantity (quantity floor), data_format: integer, value_description: commonsense evidence: The minimum quantity to enjoy the discount
        highqty      INTEGER,    ### original_column_name: highqty, column_name: high quantity , column_description: high quantity (max quantity), data_format: integer, value_description: commonsense evidence: The maximum quantity to enjoy the discount
        discount     REAL not null,    ### original_column_name: discount, column_name: discount, column_description: discount, data_format: real, value_description: 
        foreign key (stor_id)  references stores(stor_id)
                on update cascade on delete cascade
    )
    
    CREATE TABLE employee
    (
        emp_id    TEXT    ### original_column_name: emp_id, column_name: employee id, column_description: unique number identifying employees , data_format: text, value_description: 
            primary key,
        fname     TEXT not null,    ### original_column_name: fname, column_name: first name, column_description: first name of employees, data_format: text, value_description: 
        minit     TEXT,    ### original_column_name: minit, column_name: , column_description: middle name, data_format: text, value_description: 
        lname     TEXT not null,    ### original_column_name: lname, column_name: last name, column_description: last name, data_format: text, value_description: 
        job_id    INTEGER     not null,    ### original_column_name: job_id, column_name: job id, column_description: number identifying jobs, data_format: integer, value_description: 
        job_lvl   INTEGER,    ### original_column_name: job_lvl, column_name: job level, column_description: job level, data_format: integer, value_description: commonsense evidence: higher value means job level is higher
        pub_id    TEXT     not null,    ### original_column_name: pub_id, column_name: publisher id, column_description: id number identifying publishers, data_format: text, value_description: 
        hire_date DATETIME    not null,    ### original_column_name: hire_date, column_name: , column_description: hire date, data_format: datetime, value_description: 
        foreign key (job_id) references jobs(job_id)
                on update cascade on delete cascade,
        foreign key (pub_id) references publishers(pub_id)
                on update cascade on delete cascade
    )
    
    CREATE TABLE jobs
    (
        job_id   INTEGER    ### original_column_name: job_id, column_name: job id, column_description: unique id number identifying the jobs, data_format: integer, value_description: 
            primary key,
        job_desc TEXT not null,    ### original_column_name: job_desc, column_name: job description, column_description: job description, data_format: text, value_description: commonsense evidence: staff should be mentioned
        min_lvl  INTEGER     not null,    ### original_column_name: min_lvl, column_name: min level, column_description: min job level, data_format: integer, value_description: 
        max_lvl  INTEGER     not null    ### original_column_name: max_lvl, column_name: max level, column_description: max job level, data_format: integer, value_description: commonsense evidence: level range for jobs mentioned in job_desc is (min_lvl, max_lvl)
    )
    
    CREATE TABLE pub_info
    (
        pub_id  TEXT    ### original_column_name: pub_id, column_name: publication id, column_description: unique id number identifying publications, data_format: text, value_description: 
            primary key,
        logo    BLOB,    ### original_column_name: logo, column_name: , column_description: logo of publications, data_format: blob, value_description: 
        pr_info TEXT,    ### original_column_name: pr_info, column_name: publisher's information, column_description: publisher's information, data_format: text, value_description: 
        foreign key (pub_id) references publishers(pub_id)
                on update cascade on delete cascade
    )
    
    CREATE TABLE publishers
    (
        pub_id   TEXT    ### original_column_name: pub_id, column_name: publisher id, column_description: unique id number identifying publisher, data_format: text, value_description: 
            primary key,
        pub_name TEXT,    ### original_column_name: pub_name, column_name: publisher name, column_description: publisher name, data_format: text, value_description: 
        city     TEXT,    ### original_column_name: city, column_name: city, column_description: city , data_format: text, value_description: 
        state    TEXT,    ### original_column_name: state, column_name: state, column_description: state, data_format: text, value_description: 
        country  TEXT    ### original_column_name: country, column_name: country, column_description: country, data_format: text, value_description: 
    )
    
    CREATE TABLE roysched
    (
        title_id TEXT not null,    ### original_column_name: title_id, column_name: , column_description: unique id number identifying title, data_format: text, value_description: 
        lorange  INTEGER,    ### original_column_name: lorange, column_name: low range, column_description: low range, data_format: integer, value_description: 
        hirange  INTEGER,    ### original_column_name: hirange, column_name: high range, column_description: high range, data_format: integer, value_description: 
        royalty  INTEGER,    ### original_column_name: royalty, column_name: , column_description: royalty, data_format: integer, value_description: 
        foreign key (title_id)  references titles(title_id)
                on update cascade on delete cascade
    )
    
    CREATE TABLE sales
    (
        stor_id  TEXT     not null,    ### original_column_name: stor_id, column_name: store id, column_description: id number identifying stores, data_format: text, value_description: 
        ord_num  TEXT  not null,    ### original_column_name: ord_num, column_name: order number, column_description: id number identifying the orders, data_format: text, value_description: 
        ord_date DATETIME    not null,    ### original_column_name: ord_date, column_name: order date, column_description: the date of the order, data_format: datetime, value_description: 
        qty      INTEGER     not null,    ### original_column_name: qty, column_name: quantity, column_description: quantity of sales , data_format: integer, value_description: 
        payterms TEXT  not null,    ### original_column_name: payterms, column_name: , column_description: payments, data_format: text, value_description: 
        title_id TEXT   not null,    ### original_column_name: title_id, column_name: title id, column_description: id number identifying titles, data_format: text, value_description: 
        primary key (stor_id, ord_num, title_id),
        foreign key (stor_id)   references stores(stor_id)
                on update cascade on delete cascade,
        foreign key (title_id)  references titles(title_id)
                on update cascade on delete cascade
    )
    
    CREATE TABLE stores
    (
        stor_id     TEXT    ### original_column_name: stor_id, column_name: store id, column_description: unique id number of stores, data_format: text, value_description: 
            primary key,
        stor_name    TEXT,    ### original_column_name: stor_name, column_name: store name, column_description: , data_format: text, value_description: 
        stor_address TEXT,    ### original_column_name: stor_address, column_name: store address, column_description: , data_format: text, value_description: 
        city         TEXT,    ### original_column_name: city, column_name: , column_description: city name, data_format: text, value_description: 
        state        TEXT,    ### original_column_name: state, column_name: , column_description: state code, data_format: text, value_description: 
        zip          TEXT    ### original_column_name: zip, column_name: , column_description: zip code, data_format: text, value_description: 
    )
    
    CREATE TABLE titleauthor
    (
        au_id      TEXT not null,    ### original_column_name: au_id, column_name: author id, column_description: author id, data_format: text, value_description: 
        title_id   TEXT  not null,    ### original_column_name: title_id, column_name: title id, column_description: title id, data_format: text, value_description: 
        au_ord     INTEGER,    ### original_column_name: au_ord, column_name: author ordering, column_description: author ordering, data_format: integer, value_description: 
        royaltyper INTEGER,    ### original_column_name: royaltyper, column_name: , column_description: royaltyper, data_format: integer, value_description: 
        primary key (au_id, title_id),
        foreign key (au_id) references authors(au_id)
                on update cascade on delete cascade,
        foreign key (title_id)    references titles (title_id)
                on update cascade on delete cascade
    )
    
    CREATE TABLE titles
    (
        title_id  TEXT    ### original_column_name: title_id, column_name: title id, column_description: title id, data_format: text, value_description: 
            primary key,
        title     TEXT not null,    ### original_column_name: title, column_name: , column_description: title, data_format: text, value_description: 
        type      TEXT    not null,    ### original_column_name: type, column_name: , column_description: type of titles, data_format: text, value_description: 
        pub_id    TEXT,    ### original_column_name: pub_id, column_name: publisher id, column_description: publisher id, data_format: text, value_description: 
        price     REAL,    ### original_column_name: price, column_name: , column_description: price, data_format: real, value_description: 
        advance   REAL,    ### original_column_name: advance, column_name: , column_description: pre-paid amount, data_format: real, value_description: 
        royalty   INTEGER,    ### original_column_name: royalty, column_name: , column_description: royalty, data_format: integer, value_description: 
        ytd_sales INTEGER,    ### original_column_name: ytd_sales, column_name: year to date sales, column_description: year to date sales, data_format: integer, value_description: 
        notes     TEXT,    ### original_column_name: notes, column_name: , column_description: notes if any, data_format: text, value_description: commonsense evidence: had better understand notes contents and put some of them into questions if any
        pubdate   DATETIME    not null,    ### original_column_name: pubdate, column_name: publication date, column_description: publication date, data_format: datetime, value_description: 
        foreign key (pub_id) references publishers(pub_id)
                on update cascade on delete cascade
    )
}

2. Question and evidence pair samples
{
    {
        "question": "List the title, price and publication date for all sales with 'ON invoice' payment terms.",
        "evidence": "publication date refers to pubdate; payment terms refers to payterms; payterms = 'ON invoice'",
    },
    {
        "question": "What is the title that have at least 10% royalty without minimum range amount.",
        "evidence": "at least 10% royalty refers to royalty > = 10; minimum range is synonym for low range which refers to lorange; without minimum range amount refers to lorange <> 0",
    },
    {
        "question": "State the title and royalty percentage for title ID BU2075 between 10000 to 50000 range.",
        "evidence": "lorange mean low range; hirange mean high range; range refers to between the low and high range; lorange>10000; hirange<12000",
    },
    {
        "question": "Among the titles with royalty percentage, which title has the greatest royalty percentage. State it's minimum range to enjoy this royalty percentage.",
        "evidence": "minimum range is synonym for low range which refers to lorange",
    },
    {
        "question": "Provide a list of titles together with its publisher name for all publishers located in the USA.",
        "evidence": "publisher name refers to pub_name;",
    },
    {
        "question": "List all titles with sales of quantity more than 20 and store located in the CA state.",
        "evidence": "qty is abbreviation for quantity; sales of quantity more than 20 refers to qty>20; store refers to stor_name",
    },
    {
        "question": "Who are the employees working for publisher not located in USA? State the employee's name and publisher name.",
        "evidence": "not located at USA refers to country! = 'USA'",
    },
    {
        "question": "List all employees working for publisher 'GGG&G'. State their name and job description.",
        "evidence": "name = fname, lname; job description refers to job_desc; publisher refers pub_name",
    },
    {
        "question": "Among all employees, who have job level greater than 200. State the employee name and job description.",
        "evidence": "job level greater than 200 refers to job_lvl>200; job description refers to job_desc",
    },
    {
        "question": "List all the titles and year to date sales by author who are not on contract.",
        "evidence": "year to date sales refers to ytd_sales; not on contract refers to contract = 0",
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