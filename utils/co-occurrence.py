from tqdm import tqdm
import pandas as pd
from collections import Counter






def get_co_occurrence(self):

    data = mimic3sample.samples

    # Flatten the data into a DataFrame
    rows = []
    for patient in tqdm(data):
        patient_id = patient['patient_id']
        for v_idx in range(len(patient['conditions'])):
            rows.append({'patient_id': patient_id, 'codes': patient['conditions'][v_idx] + ['p'+code for code in patient['procedures'][v_idx]] + patient['drugs'][v_idx]})

    df = pd.DataFrame(rows)

    # Initialize a counter for co-occurrences
    co_occurrence_counts = Counter()

    # Count co-occurrences
    for visit in df['codes']:
        for i in range(len(visit)):
            for j in range(i + 1, len(visit)):
                co_occurrence_counts[(visit[i], visit[j])] += 1
                co_occurrence_counts[(visit[j], visit[i])] += 1

    # Create a DataFrame from the co-occurrence counts
    codes = list(set(code for visit in df['codes'] for code in visit))
    co_occurrence_matrix = pd.DataFrame(0, index=codes, columns=codes)

    for (code1, code2), count in co_occurrence_counts.items():
        co_occurrence_matrix.at[code1, code2] = count

    # Initialize the conditional probability matrix
    conditional_prob_matrix = pd.DataFrame(0.0, index=codes, columns=codes)

    # Calculate the total counts for each code
    total_counts = co_occurrence_matrix.sum(axis=1)

    # Calculate the conditional probabilities
    for code1 in codes:
        for code2 in codes:
            if total_counts[code1] > 0:
                conditional_prob_matrix.at[code1, code2] = co_occurrence_matrix.at[code1, code2] / total_counts[code1]

    codes_sorted = self.dx_table['l3'].unique().tolist() + self.rx_table['l3'].unique().tolist() + ['p' + code for code in self.px_table['l3'].unique().tolist()]
    conditional_prob_matrix_reordered = conditional_prob_matrix.reindex(index=codes_sorted, columns=codes_sorted)