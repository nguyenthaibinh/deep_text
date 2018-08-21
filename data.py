import pandas as pd

QUORA_DATA_FILE = "./datasets/quora/quora_duplicate_questions.tsv"


def load_quora_data():
	df = pd.read_csv(QUORA_DATA_FILE, sep='\t')
	text1 = list(df['question1'])
	text2 = list(df['question2'])
	is_duplicate = list(df['is_duplicate'])

	return text1, text2, is_duplicate
