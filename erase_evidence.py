import json

# 파일 경로 설정
dir = '/home/janghyeon/data/text_to_sql/DAIL-SQL/dataset/bird/'
input_file = 'dev_gold_evidence.json'
output_file = 'dev_no_evidence.json'

# JSON 파일 열기
with open(dir+input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# "evidence" 항목을 빈 문자열로 수정
for item in data:
    if 'evidence' in item:
        item['evidence'] = ""

# 수정된 데이터를 새로운 JSON 파일로 저장
with open(dir+output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"'{output_file}' file erased")

