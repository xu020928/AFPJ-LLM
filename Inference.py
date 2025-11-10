import json
import pandas as pd
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

def loadData(json_file_path, model_name, output_excel_path, output_json_path):
    try:
        # 加载模型和分词器
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 从 JSON 文件中加载数据
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_json_data = []
        t = 0

        for item in data:
            query = item.get("query")
            answer = item.get("answer")

            if query is None or answer is None:
                print(f"Skipping item due to missing query or answer: {item}")
                continue

            messages = [
                {"role": "user", "content": query}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(str(t) + ":" + response)
            t += 1

            dataset = {
                "query": query,
                "answer": answer,
                "response": response
            }
            all_json_data.append(dataset)

        # 将结果保存为 Excel 和 JSON 文件
        df = pd.DataFrame(all_json_data)
        df.to_excel(output_excel_path, index=False)
        dump_json(all_json_data, output_json_path)

    except Exception as e:
        print(f"Error occurred: {e}")

def dump_json(dataset, name):
    try:
        with open(name, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False)
    except Exception as e:
        print(f"Error occurred while saving JSON: {e}")


if __name__ == '__main__':
    filename1 = "/root/test sets/思维链（提示词完善）/output.json"
    model_name = "/root/autodl-tmp/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    output_excel_path = "/root/生成结果/基座模型生成思维链/output.xlsx"
    output_json_path = "/root/生成结果/基座模型生成思维链/new test.json"
    loadData(filename1, model_name, output_excel_path, output_json_path)