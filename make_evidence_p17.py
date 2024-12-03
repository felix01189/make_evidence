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
from sentence_transformers import SentenceTransformer
import torch
import re

openai.api_key = ""

def parse_option():
    parser = argparse.ArgumentParser("command line arguments")
    parser.add_argument("--dataset_json_path", type=str)
    parser.add_argument("--train_json_path", type=str)
    parser.add_argument("--top_n", type=int)
    parser.add_argument("--db_path", type=str)
    parser.add_argument("--train_db_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--dev_table_json_path", type=str, default="")
    parser.add_argument("--train_table_json_path", type=str, default="")
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


def read_schema_description(csv_path, db_path):
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
                            sql = f"SELECT distinct `{row[0].strip()}` FROM `{table_name}` limit 5;"
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
    def __init__(self, train_json_path, model_name="sentence-transformers/all-mpnet-base-v2"):
        with open(train_json_path, 'r') as f:
            self.train_data = json.load(f)
        
        self.questions = [item["question"] for item in self.train_data]
        self.db_ids = [item["db_id"] for item in self.train_data]
        self.evidences = [item["evidence"] for item in self.train_data]
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(self.questions, convert_to_tensor=True)
        
    def find_similar_questions(self, target_question, top_k=5):
        target_embedding = self.model.encode([target_question], convert_to_tensor=True)
        distances = torch.cdist(target_embedding, self.embeddings).squeeze(0)
        top_k_indices = torch.topk(distances, k=top_k, largest=False).indices
        top_k_questions = [(self.questions[idx], self.db_ids[idx], self.evidences[idx], distances[idx].item()) for idx in top_k_indices]
        
        return top_k_questions


def extract_evidence(response_content):
    try:
        response_json = json.loads(response_content)
        return response_json.get("evidence", "No evidence found")
    except json.JSONDecodeError:
        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
        
        matches = re.findall(json_pattern, response_content)
        
        for match in matches:
            try:
                response_json = json.loads(match)
                return response_json.get("evidence", "No evidence found")
            except json.JSONDecodeError:
                continue
        return "Response is not in JSON format"


if __name__ == "__main__":
    opt = parse_option()
    print(opt)
    with open(opt.dataset_json_path, encoding='utf-8') as f:
        question_json_all = json.load(f)
    res = []

    finder = SimilarQuestionFinder(opt.train_json_path)

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

        similar_questions = finder.find_similar_questions(data['question'], opt.top_n)
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
                response = generate_reply([{"role": "user", "content": prompt}])
            except:
                print('api error, wait for 3 seconds and retry...')
                time.sleep(3)
                pass

        data["evidence"] = extract_evidence(response)
        if opt.model == "codes":
            data["text"] = str(data["evidence"]) + " " + data["question"]
        res.append(data)
        
    with open(opt.output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
