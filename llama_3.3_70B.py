##script to execute lemma generation using Llama model family
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import transformers
import torch
import os
from huggingface_hub import login

##input your token
login(token='')
model_id = 'meta-llama/Llama-3.3-70B-Instruct'


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
file_to_predict = os.path.join(folder + '/' + language + '_' + corpus + '_target_' + file_type + '.txt')


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

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


##function that extracts the examples for a few-shot
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
    if not os.path.exists(parent_directory + '/' + folder + '/predictions_llama/'):
        os.makedirs(parent_directory + '/' + folder + '/predictions_llama/')
    with open(os.path.join(parent_directory + '/' + folder + '/predictions_llama/' + language + '_' + corpus + '_' + file_type + '_llama_predictions_' + prompt_type + '_' + input_type + '_' + run + '.tsv'),'a', encoding='utf8') as f:
        f.write("%s\n" % results_unit)

def saving_prompt(results_unit):
    if not os.path.exists(parent_directory + '/' + folder + '/prompts_llama/'):
        os.makedirs(parent_directory + '/' + folder + '/prompts_llama/')
    with open(os.path.join(parent_directory + '/' + folder + '/prompts_llama/' + language + '_' + corpus + '_' + file_type + '_llama_prompt_' + prompt_type + '_' + input_type + '_' + run + '.tsv'),'a', encoding='utf8') as f:
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
    outputs = pipeline(messages, max_new_tokens=256)
    result = outputs[0]["generated_text"][-1]
    for key, val in result.items():
        if key == 'content':
            saving_predictions_gradually(val)
            saving_predictions_gradually('')
    index +=1
    saving_prompt(messages)




