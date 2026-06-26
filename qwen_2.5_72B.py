##script to execute lemma generation using Qwen model family
#!/usr/bin/env python
# -*- coding: utf-8 -*-


import transformers
import torch
import re,os
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

login(token='')

model_name = "Qwen/Qwen2.5-72B-Instruct"


folder = '' #language folder
language = '' #language
corpus = '' #corpus name
file_type = 'test'
run = '' ##e.g. 1_run

##if we want to change the prompt type or the examples we need to look further in the corpus because
##there are some constant parts that contain this information as well
prompt_type = 'prompt_type' #e.g. basic_prompt_worst_examples_4_shot
input_type = 'wordform' 

parent_directory = 'parent_directory'
file_to_predict = os.path.join(parent_directory + folder + '/' + language + '_' + corpus + '_target_' + file_type + '.txt')

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def input_file(file):
    with open(file, 'r', encoding='utf8') as f1:
            sentence_file = f1.readlines()
    return sentence_file


def wordform_input(sentence_file):
    """
    the input to the prompt is introduced as a list of words for each sentence
    """
    fin_list = []
    for line in sentence_file:
        sentence = line.strip()
        list_of_words = sentence.split(' ')
        fin_list.append(list_of_words)
    return fin_list

def bad_examples_for_few_shot_wordform():
    with open (os.path.join(parent_directory + 'wrong_examples/' + folder + '/wrong_examples_4_shot_' + language + '.tsv'),'r', encoding='utf8') as f_examples:
        worst_cases = f_examples.readlines()
        list_wordforms = []
        list_full_example = []
        aux_wf = []
        aux_fe = []
        for line in worst_cases:
            line = line.strip()
            if line != '':
                aux_wf.append(line.split('\t')[0])
                aux_fe.append(line)
            else:
                aux_wf_str = '\n'.join(aux_wf)
                aux_fe_str = '\n'.join(aux_fe)
                list_wordforms.append(aux_wf_str)
                list_full_example.append(aux_fe_str)
                aux_wf = []
                aux_fe = []
    return list_wordforms, list_full_example


def basic_prompt_function_4_shot_wordform(folder):
    sentences, lemmatized_examples = bad_examples_for_few_shot_wordform()
    prompt = (
            f"Your task is to lemmatize a sentence in {folder}. You will be given a sentence, where each word starts from the new line. You need to provide for each word in the given sentence its dictionary form (lemma).\n"
            "For example, for the sentence:\n"
            f"{sentences[0]}\n"
            "The desired output is:\n"
            f"{lemmatized_examples[0]}\n"
            "For the sentence:\n"
            f"{sentences[1]}\n"
            "The desired output is:\n"
            f"{lemmatized_examples[1]}\n"
            "For the sentence:\n"
            f"{sentences[2]}\n"
            "The desired output is:\n"
            f"{lemmatized_examples[2]}\n"
            "For the sentence:\n"
            f"{sentences[3]}\n"
            "The desired output is:\n"
            f"{lemmatized_examples[3]}\n"            
            "Provide the output in **TSV format** (Tab-Separated Values) with the format:\n"
            "`initial word    lemma`\n"
            "Answer with the required output only, without extra spaces, quotation marks, comments or explanations.\n"
        )
    return prompt



def saving_predictions_gradually(results_unit):
    if not os.path.exists(parent_directory + '/' + folder + '/predictions_qwen/'):
        os.makedirs(parent_directory + '/' + folder + '/predictions_qwen/')
    with open(os.path.join(parent_directory + '/' + folder + '/predictions_qwen/' + language + '_' + corpus + '_' + file_type + '_qwen_predictions_72B_' + prompt_type + '_' + input_type + '_' + run + '.tsv'),'a', encoding='utf8') as f:
        f.write("%s\n" % results_unit)

def saving_prompt(results_unit):
    if not os.path.exists(parent_directory + '/' + folder + '/prompts_qwen/'):
        os.makedirs(parent_directory + '/' + folder + '/prompts_qwen/')
    with open(os.path.join(parent_directory + '/' + folder + '/prompts_qwen/' + language + '_' + corpus + '_' + file_type + '_qwen_prompt_' + prompt_type + '_' + input_type + '_' + run + '.tsv'),'a', encoding='utf8') as f:
        f.write("%s\n" % results_unit)



our_input = input_file(file_to_predict)
our_sentences = wordform_input(our_input)
our_indexes = len(our_sentences)-1
index = 0
while index <= our_indexes:
    sentence_batch = our_sentences[index]
    messages = [
    {
        "role": "system",
        "content": (basic_prompt_function_4_shot_wordform(folder))
    },
    {
        "role": "user", 
        "content": f"Sentence:\n{sentence_batch}\n"
    }
            
    ]
    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    saving_predictions_gradually(response)
    saving_predictions_gradually('')
    print(f'we are working with the sentence{index}')
    index +=1
    saving_prompt(messages)








