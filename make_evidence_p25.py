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
    parser.add_argument("--openai_api_key", type=str, default="")

    opt = parser.parse_args()

    return opt

def make_prompt(question, concat_schema):
    system_prompt = f"""### As a data science expert, you must create an evidence to assist with text-to-SQL work. 
Text-to-SQL workers receive the evidence you create as input along with db schema and question and create SQL to run on SQLite.
The evidence you create plays an important role in connecting question and correct answer SQL. 
Perform the following tasks to create an evidence that helps you create SQL.

# Step 1. Sample analysis: Prior to generating an evidence, some few-shot samples are given first. Each few-shot samples consist of question and DB scheme, for which a well-generated evidence is given as the correct answer. Analyze and understand the relationship between question, DB scheme, and evidence in each few-shot samples.
# Step 2. Problem analysis: Now, it is time to create an evidence for the problem through the relationship between question, DB scheme, and evidence understood in step 1. Understand each of the given questions and DB schema to generate evidence for a problem. And detect the word or phrase that needs evidence in the question.
# Step 3. Create evidence: Create evidence by reflecting the sample analysis of step 1 for the word or phrase detected in step 2. Describe the process of generation in detail, and produce evidence in as short a sentence as possible.
# Step 4. Print answer: Print your answers in json format of "reasoning" and "evidence".
"""

    user_prompt = f"""### problem ####################################################
1. schema of question
{{
    {concat_schema}}}
    
2. question
{{
    "question": "{question}",
    "evidence": 
}}

### Let's think step by step.

"""
    return system_prompt, user_prompt


def generate_reply(input):
    completions = openai.ChatCompletion.create(
        model="gpt-4o-mini",
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


def read_schema_description(csv_path, db_path, num_of_sampling=30):
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
    def __init__(self, train_json_all):
        embedding_model_name, self.task = "sentence-transformers/all-mpnet-base-v2", None
        # embedding_model_name, self.task = "Lajavaness/bilingual-embedding-large", None
        # embedding_model_name, self.task = "jinaai/jina-embeddings-v3", "text-matching"
        self.train_data = [item for item in train_json_all if item["evidence"].lower() not in ["", "false;"]]
        self.questions = [item["question"] for item in self.train_data]
        self.db_ids = [item["db_id"] for item in self.train_data]
        self.evidences = [item["evidence"] for item in self.train_data]
        self.model = SentenceTransformer(embedding_model_name, trust_remote_code=True, cache_folder="/home/janghyeon/data/cache")
        self.embeddings = self.model.encode(self.questions, convert_to_tensor=True, task=self.task)
        
    def find_similar_questions(self, target_question, top_k=5):
        target_embedding = self.model.encode([target_question], convert_to_tensor=True, task=self.task)
        similarities = self.model.similarity_pairwise(target_embedding, self.embeddings).squeeze(0)
        top_k_indices = torch.topk(similarities, k=top_k, largest=True).indices
        top_k_questions = [(self.questions[idx], self.db_ids[idx], self.evidences[idx]) for idx in top_k_indices]
        
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

    openai.api_key = opt.openai_api_key

    res = []
    with open(opt.dataset_json_path, encoding='utf-8') as f:
        question_json_all = json.load(f)
    with open(opt.train_json_path, encoding='utf-8') as f:
        train_json_all = json.load(f)

    print("### make evidence start  ###")
    finder = SimilarQuestionFinder(train_json_all)

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

        system_prompt, user_prompt = make_prompt(data["question"], concat_schema)

        prompt = [{"role": "system", "content": system_prompt}]

        similar_questions = finder.find_similar_questions(data['question'], opt.top_n)
        train_sample_user, train_sample_assistant, sample_num = [], [], 0
        for question, db_id, evidence in similar_questions:
            train_schema = generate_schema(f"{opt.train_db_path}/{db_id}/{db_id}.sqlite")
            sample_num += 1
            train_sample_assistant = evidence
            train_sample_user = f"""
### few-shot sample {sample_num} ####################################################
1. DB Schema of Samples
{{
{train_schema}
}}

2. Question and evidence pair samples
{{
    "question": "{question}",
    "evidence":
}}
##################################################################
"""
            prompt.append({"role": "user", "content": train_sample_user})
            prompt.append({"role": "assistant", "content": train_sample_assistant})
            
        prompt.append({"role": "user", "content": user_prompt})

        response = None
        while response is None:
            try:
                response = generate_reply(prompt)
            except:
                print('api error, wait for 3 seconds and retry...')
                time.sleep(3)
                pass

        print(response)

        data["evidence"] = str(extract_evidence(response)).replace('\n',', ')

        print(data["evidence"])

        if opt.model == "codes":
            data["text"] = data["evidence"] + " " + data["question"]
        res.append(data)
        
    with open(opt.output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
