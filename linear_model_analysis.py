import pandas as pd
from scipy.stats import linregress

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


'''
def tagging_nouns(text):
    results = pos_pipeline(text)
    nouns = [result['word'] for result in results if result['entity_group'] == "NOUN"]
    return nouns


def POS_tagging(text):
    pass
'''
data = pd.read_csv("updated_test.tsv", sep="\t")

model_name = "vblagoje/bert-english-uncased-finetuned-pos"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

pos_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_nouns_batch(words, batch_size=100):
    """
    分批处理，提取名词。
    """
    nouns = set()
    for i in range(0, len(words), batch_size):
        batch = words[i:i+batch_size]
        text = " ".join(batch)
        results = pos_pipeline(text)
        nouns.update({result['word'] for result in results if result.get('entity_group') == "NOUN"})
    return nouns

nouns = extract_nouns_batch(data['word'].tolist(), batch_size=100)
# print(nouns)


data = pd.read_csv("updated_test_word.tsv")
# 1
# filtered_data = copy.deepcopy(data) 
# 2
# filtered_data = data[(data['value'].isin(nouns))]
# 3
filtered_data = data[~((data['integrationCost'] == 0) & (data['value'].isin(nouns)))]
# 4
# filtered_data = data[(data['value'].isin(nouns)) & (data['integrationCost'] != 0)]

filtered_data.to_csv("filtered_data.csv", index=False)
y = filtered_data['meanItemRT']
x = filtered_data['integrationCost']

# y = data['meanItemRT']
# x = data['integrationCost']

slope, intercept, r_value, p_value, std_err = linregress(x, y)

print("Linear Regression Results:")
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"P-value: {p_value}")

