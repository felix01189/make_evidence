set -e

dataset_json_path="../codes/data/sft_bird_with_evidence_dev_text2sql.json"
train_json_path="../codes/data/sft_bird_with_evidence_train_text2sql.json"
top_k=5
db_path="../codes/data/sft_data_collections/bird/dev/dev_databases"
train_db_path="../codes/data/sft_data_collections/bird/train/train_databases"
output_path="../codes/data/sft_bird_with_my_evidence_dev_text2sql_prompt31_3.json"
openai_api_key=`cat openai_api_key`

python make_evidence_p31.py \
--dataset_json_path $dataset_json_path \
--train_json_path $train_json_path \
--top_k $top_k \
--db_path $db_path \
--train_db_path $train_db_path \
--output_path $output_path \
--openai_api_key $openai_api_key
