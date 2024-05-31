from datasets import load_dataset
from tabulate import tabulate

from fmeval.eval_algorithms.qa_accuracy import QAAccuracy, QAAccuracyConfig

import random
import boto3
import botocore
from botocore.client import Config
import json
import pandas as pd

import plotly.express as px

def huggingFaceDatasetDownloader(download_file_path):  
    # Load the dataset from HuggingFace
    dataset = load_dataset("databricks/databricks-dolly-15k")
    df = dataset["train"].to_pandas()

    # Display the first records
    #print(tabulate(df.head(1), headers='keys', tablefmt='psql'))

    record_count = len(df)
    print("Record count:", record_count)

    # Group by category column and display the count for each category
    category_counts = df.groupby("category").size().reset_index(name='count')
    print("Category Counts:")  
    print(tabulate(category_counts, headers='keys', tablefmt='psql'))

    # Save the DataFrame as a CSV file
    df.to_csv(download_file_path, index=False)
    return df

def invokeMetaLlama3Model(df, random_sample_count): 
    random_records = random.sample(range(len(df)), random_sample_count)
    df_sample = df.iloc[random_records].copy()
    df_sample['prompt'] = ""
    df_sample['metaLlama3Response'] = ""

    bedrock_runtime = boto3.client('bedrock-runtime', config=Config(read_timeout=500))

    count = 0
    for index, row in df_sample.iterrows():
        count += 1
        print(f"Processing the request for Llama3 Model: {count}")
        instruction = row['instruction']
        context = row['context']
        category = row['category']

        prompt = f"""
                 Category: {category}
                 Instruction: {instruction}
                 Context: {context}

                 Please provide a precise response to the instruction and context based on the category.

                 """
        try:
            body = json.dumps({"prompt": prompt, "max_gen_len":200, "temperature":0.5, "top_p":0.9})
            modelId = "meta.llama3-70b-instruct-v1:0"
            accept = "application/json"
            contentType = "application/json"

            response = bedrock_runtime.invoke_model(
                body=body, modelId=modelId, accept=accept, contentType=contentType
            )
            response_body = json.loads(response.get("body").read()).get("generation")

            #print(response_body)
            df_sample.at[index, 'prompt'] = prompt
            df_sample.at[index, 'metaLlama3Response'] = response_body

        except botocore.exceptions.ClientError as error:
            if error.response['Error']['Code'] == 'AccessDeniedException':
                   print(f"\x1b[41m{error.response['Error']['Message']} \
                        \nTo troubeshoot this issue please refer to the following resources.\
                         \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                         \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")

            else:
                raise error
    return df_sample

def invokeAnthropicModel(df_sample, response_file_path):
    df_sample['anthropicResponse'] = ""

    bedrock_runtime = boto3.client('bedrock-runtime', config=Config(read_timeout=500))

    count = 0
    for index, row in df_sample.iterrows():
        count += 1
        print(f"Processing the request for Anthropic Model: {count}")
        instruction = row['instruction']
        context = row['context']
        category = row['category']

        prompt = f"""
                 Category: {category}
                 Instruction: {instruction}
                 Context: {context}

                 Please provide a precise response to the instruction and context based on the category.

                 """
        try:
            messages=[{ "role":'user', "content":[{'type':'text','text': prompt}]}]
            body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 200, "messages": messages, "temperature": 0.5, "top_p": 0.9})
            modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
            accept = "application/json"
            contentType = "application/json"

            response = bedrock_runtime.invoke_model(
                body=body, modelId=modelId, accept=accept, contentType=contentType
            )
            response_body = json.loads(response.get('body').read())
            response_text = response_body.get('content')[0]['text']

            #print(response_text)
            df_sample.at[index, 'anthropicResponse'] = response_text

        except botocore.exceptions.ClientError as error:
            if error.response['Error']['Code'] == 'AccessDeniedException':
                   print(f"\x1b[41m{error.response['Error']['Message']}\
                        \nTo troubeshoot this issue please refer to the following resources.\
                         \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                         \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")

            else:
                raise error
    df_sample.to_json(response_file_path, orient='records', lines="true")
    return df_sample


def modelEvaluator(model_name, model_output_attribute, model_output_file, evaluator_response_file):
    config = DataConfig(
                dataset_name = model_name,
                dataset_uri = model_output_file,
                dataset_mime_type = MIME_TYPE_JSONLINES,
                model_input_location = "instruction",
                target_output_location = "response",
                model_output_location = model_output_attribute
        )

    # Configure and run QAAccuracy evaluation
    qa_eval = QAAccuracy(QAAccuracyConfig(target_output_delimiter="<OR>"))
    results = qa_eval.evaluate(dataset_config=config, save=True)
    #print(json.dumps(results, default=vars, indent=4))
    with open(evaluator_response_file, 'w') as f:
        json.dump(results, f, default=lambda c: c.__dict__)
        print(f'Results saved to {evaluator_response_file}')
    return results


def load_results(model_names, evaluator_response_folder):
    accuracy_results = []
    for model_name in model_names:
        file = f'{evaluator_response_folder}/{model_name}.json'
        with open(file, 'r') as f:
            res = json.load(f)
            for accuracy_eval in res:
                for accuracy_scores in accuracy_eval["dataset_scores"]:
                    accuracy_results.append(
                        {'model': model_name, 'evaluation': 'accuracy', 'dataset': accuracy_eval["dataset_name"],
                         'metric': accuracy_scores["name"], 'value': accuracy_scores["value"]})

    accuracy_results_df = pd.DataFrame(accuracy_results)
    return accuracy_results_df


def visualize_radar(results_df, plotfilePath, dataset):
    if dataset == 'all':
       mean_across_datasets = results_df.drop('evaluation', axis=1).groupby(['model', 'metric']).describe()['value']['mean']
       results_df = pd.DataFrame(mean_across_datasets).reset_index().rename({'mean':'value'}, axis=1)
    else:
        results_df = results_df[results_df['dataset'] == dataset]

    fig = px.line_polar(results_df, r='value', theta='metric', color='model', line_close=True) 
    xlim = 1
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, xlim],
            )),
        margin=dict(l=150, r=0, t=100, b=80)
    )

    title = 'Average Performance over databricks/databricks-dolly-15k' if dataset == 'all' else dataset
    fig.update_layout(
            title=dict(text=title, font=dict(size=20), yref='container')
        )
    fig.show()
    fig.write_image(plotfilePath)


def main():
    user_dir = "/home/sagemaker-user/"
    models = ["Meta_Llama3_70b_Instruct", "Anthropic_Claude_3_Sonnet"]
    random_sample_count = 3000
    
    df = huggingFaceDatasetDownloader(user_dir + "databricks-dolly-15k.csv")
    df_sample = invokeMetaLlama3Model(df, random_sample_count)
    df_sample = invokeAnthropicModel(df_sample, user_dir + "response.json")
    modelEvaluator(models[0], "metaLlama3Response", user_dir + "response.json", user_dir + f"{models[0]}.json")
    modelEvaluator(models[1], "anthropicResponse", user_dir + "response.json", user_dir + f"{models[1]}.json")

    results_df = load_results(models, user_dir)
    visualize_radar(results_df, user_dir + "modelEvaluvationPlot.pdf", dataset='all')

if __name__ == '__main__':
    main()
