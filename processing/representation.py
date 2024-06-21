import numpy as np
import itertools
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from multiprocessing import Pool
import time
from numba import jit
import numpy as np
import pandas as pd
#from mpi4py import MPI
import os
import csv
import sklearn
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
from collections import defaultdict
from keras.utils import Sequence

class DNA_FULL:
    @staticmethod
    def one_hot_encoding(sequences, max_length=100):
        """
        One-hot encode a list of DNA sequences.

        Parameters:
        sequences (list of str): List of DNA sequences.
        max_length (int): Maximum length of the sequences. Sequences longer than this will be truncated,
                          and sequences shorter than this will be padded with 'N'.

        Returns:
        np.ndarray: A 3D numpy array of shape (num_sequences, max_length, 4) representing the one-hot encoded sequences.
        """
        # Define a dictionary to map nucleotides to integers
        nucleotide_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

        # Initialize an empty array for the one-hot encoded sequences
        one_hot_encoded = np.zeros((len(sequences), max_length, 4), dtype=int)

        for i, sequence in enumerate(sequences):
            # Truncate or pad the sequence to the maximum length
            sequence = sequence[:max_length].ljust(max_length, 'N')
            for j, nucleotide in enumerate(sequence):
                if nucleotide in nucleotide_to_int and nucleotide != 'N':
                    one_hot_encoded[i, j, nucleotide_to_int[nucleotide]] = 1
        return one_hot_encoded
        
    @staticmethod
    def kmer_count(sequence, domaine="ATCG", k=3, step=1):
        """
        Utils: to count kmer occurrence in DNA sequence and compute frequency.
        """
        start_time = time.time()  # Start the timer
        kmers = [''.join(p) for p in itertools.product(domaine, repeat=k)]
        kmers_count = {kmer: 0 for kmer in kmers}
        s = 0
        
        for i in range(0, len(sequence) - k + 1, step):
            kmer = sequence[i:i + k]
            s += 1
            if kmer in kmers:
                kmers_count[kmer] += 1
        for key in kmers_count:
            kmers_count[key] = kmers_count[key] / s
            
        end_time = time.time()
        elapsed_time = end_time - start_time        
        return kmers_count
    
    @staticmethod
    def build_kmer_representation(df, domaine="ATCG", k=3, dtypes=['float64', 'int64'], asCudaDF=False):
        """
        Utils: For given k-mer generate dataset and return vectorised version
        """
        # Count
        sequences   = df['sequence']
        kmers_count = np.array([DNA_FULL.kmer_count(sequence, domaine, k=k, step=1) for sequence in sequences])
        
        # Vectorize
        v = DictVectorizer(sparse=False)
        feature_values = v.fit_transform(kmers_count)
        feature_names = v.get_feature_names_out()

        # dtypes and save df
        dtype = {col: dtypes[0] for col in feature_names}
        X = pd.DataFrame(feature_values, columns=feature_names).astype(dtype)
        y = df['class'].astype(dtypes[1])

        
        if asCudaDF:
            import cudf 
            X_cuda = cudf.DataFrame.from_pandas(X)
            y_cuda = cudf.Series(y)
            return X_cuda, y_cuda, feature_names
            
        return X, y, feature_names

class DNA_MPI:
    @staticmethod
    def kmer_count(sequence, domaine="ATCG", k=3, step=1):
        """
        Utils: to count kmer occurrence in DNA sequence and compute frequency.
        """
        start_time = time.time()  # Start the timer
        kmers = [''.join(p) for p in itertools.product(domaine, repeat=k)]
        kmers_count = {kmer: 0 for kmer in kmers}
        s = 0
        
        for i in range(0, len(sequence) - k + 1, step):
            kmer = sequence[i:i + k]
            s += 1
            kmers_count[kmer] += 1
        for key in kmers_count:
            kmers_count[key] = kmers_count[key] / s
            
        end_time = time.time()
        elapsed_time = end_time - start_time        
        return kmers_count
    
    @staticmethod
    def build_kmer_representation(df, domaine="ATCG", k=3, multiprocess=False, workers=4, output_file='./Content/Data/kmer_sample.csv'):
        """
        Utils: For given k-mer generate dataset and return vectorized version or
               MPI version to count kmer occurrence in DNA sequences and save a vectorized
               representation to a CSV file.
        """
        if not multiprocess:
            sequences = df['sequence']
            kmers_count = np.array([DNA_MPI.kmer_count(sequence, domaine, k=k, step=1) for sequence in sequences])
            v = DictVectorizer(sparse=False)
            feature_values = v.fit_transform(kmers_count)
            feature_names = v.get_feature_names_out()
            X = pd.DataFrame(feature_values, columns=feature_names)
            y = df['class']
            return X, y
        else:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            
            if rank == 0:
                # Master process
                start_time = time.time()
                chunk_size = len(df) // workers
                chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(workers)]

                if len(df) % workers != 0:
                    chunks[-1] = pd.concat([chunks[-1], df.iloc[workers*chunk_size:]])

                # Save the chunks as temporary CSV files
                os.makedirs('temp_chunks', exist_ok=True)
                for i, chunk in enumerate(chunks):
                    chunk.to_csv(f'temp_chunks/chunk_{k}_{i}.csv', index=False)
                
                # Create empty output files for each worker
                for i in range(workers):
                    with open(f'temp_chunks/output_{i}.csv', 'w') as f:
                        pass
    
            # Synchronize all processes
            comm.Barrier()
            if rank < workers:
                chunk_df = pd.read_csv(f'temp_chunks/chunk_{k}_{rank}.csv')
                sequences = chunk_df['sequence']
                targets = chunk_df['class']

                # Compute k-mer counts
                kmer_dicts = []
                for i, sequence in enumerate(sequences):
                    kmer_dict = DNA.kmer_count(sequence, domaine, k=k, step=1)
                    kmer_dict['class'] = targets.iloc[i]
                    kmer_dicts.append(kmer_dict)

                # Save k-mer counts to CSV
                temp_output_file = f'temp_chunks/output_{rank}.csv'
                with open(temp_output_file, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=list(kmer_dicts[0].keys()))
                    writer.writeheader()
                    writer.writerows(kmer_dicts)

            # Synchronize all processes
            comm.Barrier()
            if rank == 0:
                with open(output_file, mode='w', newline='') as final_output:
                    writer = None
                    for i in range(workers):
                        temp_output_file = f'temp_chunks/output_{i}.csv'
                        with open(temp_output_file, mode='r') as file:
                            reader = csv.DictReader(file)
                            if writer is None:
                                writer = csv.DictWriter(final_output, fieldnames=reader.fieldnames)
                                writer.writeheader()
                            for row in reader:
                                writer.writerow(row)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"[workers={workers}]\t - Execution time: {elapsed_time:.6f} seconds")
                shutil.rmtree('temp_chunks')
                return True
                return pd.read_csv(output_file)
                
class DNA:  
    @staticmethod
    def kmer_count_v2(sequence, domaine="ACDEFGHIKLMNPQRSTVWYX", k=3, step=1):
        """
        Utils: to count kmer occurrence in DNA sequence and compute frequency
        """
        kmers_count = defaultdict(int)
        total_kmers = 0
        for i in range(0, len(sequence) - k + 1, step):
            kmer = sequence[i:i + k]
            kmers_count[kmer] += 1
            total_kmers += 1
        for key in kmers_count:
            kmers_count[key] /= total_kmers
        return kmers_count

    @staticmethod
    def build_kmer_representation_v2(df, domaine="ACDEFGHIKLMNPQRSTVWYX", k=3, dtypes=['float64', 'int64'], asCudaDF=False, batch_size=1000, feature_mask=None):
        """
        Utils: For given k-mer generate dataset and return vectorized version
        """
        sequences = df['sequence']
        y = df['class']#.astype(dtypes[1])
        
        # Initialize DictVectorizer
        v = DictVectorizer(sparse=True)
        kmers_count_list = []
        
        # Process in batches
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i + batch_size]
            batch_kmers_count = [DNA.kmer_count_v2(sequence, domaine, k=k, step=1) for sequence in batch_sequences]
            kmers_count_list.extend(batch_kmers_count)
        
        # Vectorize the kmer counts
        feature_values = v.fit_transform(kmers_count_list)
        feature_names = v.get_feature_names_out()
        
        # Convert to DataFrame
        X = pd.DataFrame.sparse.from_spmatrix(feature_values, columns=feature_names)#.astype(dtypes[0])
        X = X.sparse.to_dense()

        # Apply feature mask if provided
        if feature_mask is not None:
            # Ensure feature_mask is a set for quick lookup
            feature_mask_set = set(feature_mask)
            current_features = set(X.columns)
            for feature in feature_mask_set - current_features:
                X[feature] = 0
            X = X[feature_mask]
        
        if asCudaDF:
            import cudf
            X_cuda = cudf.DataFrame.from_pandas(X)
            y_cuda = cudf.Series(y)
            return X_cuda, y_cuda, feature_names
        
        return X, y, feature_names

class DataGenerator(Sequence):
    def __init__(self, dataset, feature_mask, batch_size=64, shuffle=True, domaine=None, k=None):
        self.dataset = dataset
        self.feature_mask = feature_mask
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.domaine = domaine
        self.k = k
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.dataset) / self.batch_size))
    
    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch, y_batch, _ = DNA.build_kmer_representation_v2(
            self.dataset[indices], feature_mask=self.feature_mask, domaine=self.domaine, k=self.k, 
            dtypes=['float16', 'int8'], asCudaDF=False
        )
        return X_batch, y_batch
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)
