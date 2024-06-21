import numpy as np
import pandas as pd

def read(file_path, family, domaine="ACDEFGHIKLMNPQRSTVWY"):
    """
    Utils: to read fasta file and extract content as dataframe, replacing any character not in the specified domaine with 'X'.
    """
    sequences = []
    with open(file_path, 'r') as file:
        current_id = None
        current_sequence = ''
        for line in file:
            if line.startswith('>'):
                if current_id:
                    sequences.append({
                        'id': current_id,
                        'sequence': ''.join([char if char in domaine else 'X' for char in current_sequence]),
                        'length': len(current_sequence),
                        'class': family
                    })
                current_id = line.strip().split('|')[0][1:].strip()
                current_sequence = ''
            else:
                current_sequence += line.strip()
        if current_id:
            sequences.append({
                'id': current_id,
                'sequence': ''.join([char if char in domaine else 'X' for char in current_sequence]),
                'length': len(current_sequence),
                'class': family
            })
    
    df = pd.DataFrame(sequences)
    return df
    
def read_fasta_with_family(file_path, default_family):
    """
    Utils: to read fasta file and extract content as dataframe.
    If the family is specified in the header, it will be used; otherwise, the default_family will be used.
    """
    sequences = []
    family_pattern = re.compile(r'\|([^\|]+)\|([^\|]+)$')

    with open(file_path, 'r') as file:
        current_id = None
        current_sequence = ''
        current_family = default_family
        for line in file:
            if line.startswith('>'):
                if current_id:
                    sequences.append({'id': current_id, 'sequence': current_sequence, 'length': len(current_sequence), 'class': current_family})
                header_parts = line.strip().split('|')
                current_id = header_parts[0][1:].strip()
                family_match = family_pattern.search(line.strip())
                if family_match:
                    current_family = family_match.group(2).strip()
                else:
                    current_family = default_family
                current_sequence = ''
            else:
                current_sequence += line.strip()
        if current_id:
            sequences.append({'id': current_id, 'sequence': current_sequence, 'length': len(current_sequence), 'class': current_family})

    df = pd.DataFrame(sequences)
    return df