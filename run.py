import pandas as pd
import math
import transformers
import torch
import copy


file_a_path = 'naturalstories/parses/stanford/all-parses-aligned.txt.stanford'
file_b_path = 'naturalstories/parses/stanford/all-parses.txt.stanford'

# Load data from files
data_a = pd.read_csv(file_a_path, sep='\t', header=None, names=['Dependency_A'])
data_b = pd.read_csv(file_b_path, sep='\t', header=None, names=['Dependency_B'])

def calculate_ic(dependency):
    relation, content = dependency.split("(", 1)
    head, dependent = content.strip(")").split(", ")
    head_pos = int(head.split("-")[-1])
    dependent_pos = int(dependent.split("-")[-1])
    ic = abs(dependent_pos - head_pos + 1)

    if ic:
       ic = math.log(ic)
    
    # Add 1 to IC if the relation is ccomp or nsubj
    
    # ic = dependent_pos - head_pos
    # if relation in {"ccomp", "nsubj", "acl", "dobj", "nmod"}: # acl:relcl, ccomp,dobj, nsubj:pass, and nmod
        # ic += 1
    # ic = abs(ic)
    
    return ic

data_b['integrationCost'] = data_b['Dependency_B'].apply(calculate_ic)

def parse_a_format(dependency_a):
    _, content = dependency_a.split("(", 1)
    _, dependent = content.strip(")").split(", ")
    zone_info = dependent.split("-")[-1]
    if "." in zone_info:
        parts = zone_info.split(".")
        item = parts[0]
        zone = parts[1]
    else:
        item = zone_info
        zone = zone_info
    return int(item), int(zone)

data_a[['item', 'zone']] = data_a['Dependency_A'].apply(lambda x: pd.Series(parse_a_format(x)))

integration_cost_result = pd.concat([data_a[['item', 'zone']], data_b[['integrationCost']]], axis=1)

output_path = 'result.csv'
integration_cost_result.to_csv(output_path, sep='\t', index=False)

file_wordinfo_path = "naturalstories/naturalstories_RTS/processed_wordinfo.tsv"  

integration_cost_data = integration_cost_result  
wordinfo_data = pd.read_csv(file_wordinfo_path, sep="\t")  

# Merge the files on 'zone' and 'item'
merged_data = pd.merge(integration_cost_data, wordinfo_data, on=["zone", "item"], how="inner")


merged_data = merged_data[["item", "zone", "integrationCost", "meanItemRT"]]

output_path = "merged_data.csv"
merged_data.to_csv(output_path, index=False)


test2 = pd.read_csv("naturalstories/words.tsv", sep="\t", header=None, names=["key", "value"])


merged_data_whole = copy.deepcopy(merged_data)
merged_data_word = copy.deepcopy(merged_data)

merged_data_whole['key'] = merged_data_whole['item'].astype(str) + "." + merged_data_whole['zone'].astype(str) + ".whole"
merged_data_word['key'] = merged_data_word['item'].astype(str) + "." + merged_data_word['zone'].astype(str) + ".word"

merged_whole = pd.merge(merged_data_whole, test2, how="left", on="key")
merged_word = pd.merge(merged_data_word, test2, how="left", on="key")

if 'word' not in merged_whole:
    merged_whole['word'] = ""
if 'word' not in merged_word:
    merged_word['word'] = ""


merged_word.to_csv("updated_test_word.tsv", index=False)
