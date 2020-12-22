from torch.utils.data import Dataset
import torch
import pandas as pd
from tqdm import tqdm


class NarrativeQASummariesDataset(Dataset):
    def __init__(self, questions_file, summaries_file):
        # create df for both
        self.questions_df = pd.read_csv(questions_file)
        self.summaries_df = pd.read_csv(summaries_file)

    def __len__(self):
        # return the len of the number of documents --> len of summaries_df
        return len(self.summaries_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # items from summaries
        document_id = self.summaries_df.iloc[idx]['document_id']
        summary = self.summaries_df.iloc[idx]['summary']

        # items from questions
        questions = []
        answers = []  # stored as tuple with answer1,answer2
        for i in range(len(self.questions_df)):
            curr_row = self.questions_df.iloc[i]
            if curr_row['document_id'] == document_id:
                questions.append(curr_row['question'])
                curr_answers = (curr_row['answer1'], curr_row['answer2'])
                answers.append(curr_answers)

        sample = {'summary': summary, 'questions': questions,
                  'answers': answers, 'idx': idx}
        return sample


# testing dataset (just an example)
questions_file = 'narrativeqa-master/qaps.csv'
summaries_file = 'narrativeqa-master/third_party/wikipedia/summaries.csv'
dataset = NarrativeQASummariesDataset(
    questions_file=questions_file, summaries_file=summaries_file)
dataset_global = list()
for i, elem in tqdm(enumerate(dataset), total=len(dataset)):
    for j, q in enumerate(elem['questions']):
        obj = [i, q, elem['answers'][j]]
        dataset_global.append(obj)

print(dataset_global[0:10])

pd.DataFrame(dataset_global).to_csv(
    'NarrativeQA_map.csv', index=False, header=False)
