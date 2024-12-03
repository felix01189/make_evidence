import json
import argparse
import openai
from tqdm import tqdm
import sqlite3
import os
import time
import csv
from sentence_transformers import SentenceTransformer
import torch
import re
import jellyfish

openai.api_key = ""

def parse_option():
    parser = argparse.ArgumentParser("command line arguments")
    parser.add_argument("--dataset_json_path", type=str)
    parser.add_argument("--train_json_path", type=str)
    parser.add_argument("--top_n", type=int)
    parser.add_argument("--db_path", type=str)
    parser.add_argument("--train_db_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--dev_table_json_path", type=str)
    parser.add_argument("--train_table_json_path", type=str)
    parser.add_argument("--model", type=str, default="codes")

    opt = parser.parse_args()

    return opt

def make_prompt(question, concat_schema, train_sample):
    prompt = f"""### You are an assistant to a data scientist. 
Your objective is to generate evidence by understanding the given question, database schema, database column descriptions. 

### Follow the instructions below:
# 1 - Read the samples and understand the relationship between question and database schema and descriptions.
# 2 - Read the question of problem.
# 3 - Analyze the database schema, database column descriptions and database value samples of problem. 
# 4 - Generate evidence so that it is as short as possible and contains as much information as possible.
# 5 - Answer in json format. Format instructions: "reasoning", "evidence".

{train_sample}

### problem ####################################################
"""
    prompt += f"""1. schema of question
{{
    {concat_schema}}}
    
2. question
{{
    "question": "{question}",
    "evidence": 
}}

### Let's think step by step.

"""
    return prompt

def make_keyword_erase_prompt(question, schema_list, value_list):
    prompt = f"""### Objective: Analyze the given question to identify and erase schemas and value samples. 
These elements are crucial for comparing the structure of sentences.

### Instructions:
1. Read the question carefully and understand the structure of question.
2. Read the list of schema_list:
  Read schema_list carefully and search for the schema names in the schema_list are included in the question. 
  The schema names may not necessarily be the same, so find schema names that are semantically similar.
3. Read the list of value_list:
  Perform the same task as number 2 for value_list
4. Erase schemas and value samples: 
  Change the schema name to "<schema>".
  Change the value sample to "<value>".
5 - Answer in json format. Format instructions: "reasoning", "masked_question".

### Task:
  question: {question}
  schema_list: {schema_list}
  value_list: {value_list}

### Let's think step by step.

"""
    return prompt


def generate_reply(input, model_name="gpt-4o-mini"):
    completions = openai.ChatCompletion.create(
        model=model_name,
        messages=input,
        temperature=0.
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


def read_schema_description(csv_path, db_path, num_of_sampling=10):
    files = sorted(os.listdir(csv_path))
    schema_description = ""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for csv_file in files:
        file_path = os.path.join(csv_path, csv_file)
        table_name = os.path.splitext(csv_file)[0]  

        try:
            with open(file_path, 'r', encoding='cp1252') as f:
                lines = [line.replace('\x00', '').replace('\n', ' ').replace('\r', ' ').strip() for line in f if line.strip()]

                csv_reader = csv.reader(lines)
                headers = next(csv_reader)

                for row in csv_reader:
                    if row:
                        row_description = ", ".join([f"{header}: {value}" for header, value in zip(headers, row)])
                        row_description = "   ### " + row_description
                        schema_description += row_description

                        try:
                            # sql = f"SELECT distinct `{row[0].strip()}` FROM `{table_name}` limit {num_of_sampling};"
                            sql = f"SELECT distinct `{row[0].strip()}` FROM (SELECT `{row[0].strip()}` FROM `{table_name}` limit 1000) limit {num_of_sampling};"
                            cursor.execute(sql)
                            column_values = cursor.fetchall()
                            column_value = ""
                            for value in column_values:
                                column_value += (str(value[0]) + ", ")
                            schema_description = schema_description + "   ### column value examples: " + column_value + '\n'
                        except Exception as e:
                            schema_description += '\n'

        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    conn.close()
    return schema_description


class SimilarQuestionFinder:
    def __init__(self, train_json_all, model_name):
        self.train_data = [item for item in train_json_all if item["evidence"].lower() not in ["", "false;"]]
        self.questions = [item["question"] for item in self.train_data]
        self.masked_questions = [item["masked_question"] for item in self.train_data]
        self.db_ids = [item["db_id"] for item in self.train_data]
        self.evidences = [item["evidence"] for item in self.train_data]
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(self.masked_questions, convert_to_tensor=True)
        
    def find_similar_questions(self, target_question, top_k=5):
        target_embedding = self.model.encode([target_question], convert_to_tensor=True)
        distances = torch.cdist(target_embedding, self.embeddings).squeeze(0)
        top_k_indices = torch.topk(distances, k=top_k, largest=False).indices
        top_k_questions = [(self.questions[idx], self.db_ids[idx], self.evidences[idx], distances[idx].item()) for idx in top_k_indices]

        return top_k_questions

def extract_json_item(item_name, response_content):
    try:
        response_json = json.loads(response_content)
        return response_json.get(item_name, f"No {item_name} found")
    except json.JSONDecodeError:
        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
        
        matches = re.findall(json_pattern, response_content)
        
        for match in matches:
            try:
                response_json = json.loads(match)
                return response_json.get(item_name, f"No {item_name} found")
            except json.JSONDecodeError:
                continue
        return "Response is not in JSON format"


def sampling_database_value(db_folder_path, table_json, num_of_sampling=10):
    
    for data in table_json:
        value_samples = []  
        db_id = data['db_id']
        tables = data['table_names_original']
    
        conn = sqlite3.connect(f"{db_folder_path}/{db_id}/{db_id}.sqlite")
        cursor = conn.cursor()

        for table in tables:
            try:
                sql = f"SELECT name FROM PRAGMA_TABLE_INFO('{table}');"
                cursor.execute(sql)
                column_info = cursor.fetchall()
                column_names = [column[0] for column in column_info]

                for column_name in column_names:
                        try:
                            # sql = f"SELECT distinct `{column_name}` FROM `{table}` limit {num_of_sampling};"
                            sql = f"SELECT distinct `{column_name}` FROM (SELECT `{column_name}` FROM `{table}` limit 1000) limit {num_of_sampling};"
                            cursor.execute(sql)
                            value_info = cursor.fetchall()
                            value_list = [value[0] for value in value_info]
                            value_samples += value_list
                        except Exception as e:
                            print(f"Error reading column {column_name} of table {table}: {e}")

            except Exception as e:
                print(f"Error reading table {table}: {e}")
            
            data['value_samples'] = value_samples
    
        conn.close()
    return table_json

def calculate_edit_distance(word1, word2, threshold=0.9):
    processed_word1, processed_word2 = str(word1).lower(), str(word2).lower()
    return jellyfish.jaro_winkler_similarity(processed_word1, processed_word2) >= threshold

def mask_similar_words(question, reference_list, mask_string):
    masked_words = question.split().copy()
    
    for i, target_word in enumerate(masked_words):
        for ref_setence in reference_list:
            ref_words = str(ref_setence).split().copy()
            if str(target_word).lower() not in ["what","how","when","which","when","where",\
                                "in","the","for","that",\
                                "is","are",\
                                "min","max","count","sum","average"]:
                for ref_word in ref_words:
                    if calculate_edit_distance(target_word, ref_word):
                        masked_words[i] = mask_string
                        break
    
    return re.sub(rf'({mask_string})(\s+{mask_string})+', r'\1', " ".join(masked_words))

def question_masking(json_data, table_json):
    for data in json_data:
        db_id = data['db_id']
        question = data['question']

        schema_list, value_list = [], []
        for item in table_json:
            if item['db_id'] == db_id:
                schema_list += item['table_names']
                schema_list += item['table_names_original']
                for column_name in item['column_names']:
                    schema_list.append(column_name[1])
                for column_name_original in item['column_names_original']:
                    schema_list.append(column_name_original[1])
                value_list += item['value_samples']

        prompt = make_keyword_erase_prompt(question, schema_list, value_list)

        response = None
        while response is None:
            try:
                response = generate_reply([{"role": "user", "content": prompt}], model_name="gpt-4o-mini")
            except Exception as e:
                print(e)
                time.sleep(3)
                pass

        masked_question = extract_json_item("masked_question", response)
        data['masked_question'] = masked_question
        
        print(question)
        print(masked_question)
    return json_data




if __name__ == "__main__":
    opt = parse_option()
    print(opt)
    res = []

    print("### table json process ###")
    with open(opt.dev_table_json_path, encoding='utf-8') as f:
        dev_table_json = json.load(f)
    with open(opt.train_table_json_path, encoding='utf-8') as f:
        train_table_json = json.load(f)
    dev_table_json = sampling_database_value(opt.db_path, dev_table_json) 
    train_table_json = sampling_database_value(opt.train_db_path, train_table_json) 

    print("### question json process ###")
    with open(opt.dataset_json_path, encoding='utf-8') as f:
        question_json_all = json.load(f)
    with open(opt.train_json_path, encoding='utf-8') as f:
        train_json_all = json.load(f)
    question_json_all = question_masking(question_json_all, dev_table_json)
    train_json_all = question_masking(train_json_all, train_table_json)

    print("### make evidence start  ###")
    model_name = "sentence-transformers/all-mpnet-base-v2"
    finder = SimilarQuestionFinder(train_json_all, model_name)

    for i, data in enumerate(tqdm(question_json_all)):
        schema = generate_schema(f"{opt.db_path}/{data['db_id']}/{data['db_id']}.sqlite")
        schema_description = read_schema_description(f"{opt.db_path}/{data['db_id']}/database_description", f"{opt.db_path}/{data['db_id']}/{data['db_id']}.sqlite")

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

        similar_questions = finder.find_similar_questions(data['masked_question'], opt.top_n)
        train_sample, sample_num = "", 0
        for question, db_id, evidence, distance in similar_questions:
            train_schema = generate_schema(f"{opt.train_db_path}/{db_id}/{db_id}.sqlite")
            sample_num += 1
            train_sample += f"""
### sample {sample_num} ####################################################
1. DB Schema of Samples
{{
{train_schema}
}}

2. Question and evidence pair samples
{{
    "question": "{question}",
    "evidence": "{evidence}"
}}
##################################################################
"""

        prompt = make_prompt(data["question"], concat_schema, train_sample)

        response = None
        while response is None:
            try:
                response = generate_reply([{"role": "user", "content": prompt}], model_name="gpt-4o-mini")
            except:
                print('api error, wait for 3 seconds and retry...')
                time.sleep(3)
                pass

        data["evidence"] = extract_json_item("evidence", response)
        if opt.model == "codes":
            data["text"] = str(data["evidence"]) + " " + data["question"]
        res.append(data)
        
    with open(opt.output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
