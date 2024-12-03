set -e

dataset_json_path="/home/janghyeon/data/text_to_sql/DAIL-SQL/dataset/bird/dev/dev.json"
train_json_path="/home/janghyeon/data/text_to_sql/DAIL-SQL/dataset/bird/train/train.json"
top_n=3
db_path="/home/janghyeon/data/text_to_sql/DAIL-SQL/dataset/bird/database"
train_db_path="/home/janghyeon/data/text_to_sql/DAIL-SQL/dataset/bird/database"
output_path="/home/janghyeon/data/text_to_sql/DAIL-SQL/dataset/bird/dev/dev_with_my_evidence.json"
model="DAIL-SQL"

python make_evidence.py \
--dataset_json_path $dataset_json_path \
--train_json_path $train_json_path \
--top_n $top_n \
--db_path $db_path \
--train_db_path $train_db_path \
--output_path $output_path \
--model $model
