import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import json
from Bio import SeqIO
import sys
current_directory = os.getcwd()
root_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
utils_directory = os.path.join(root_directory, 'processing')
sys.path.append(utils_directory)

import fasta
from representation import DNA

class SingleKModel:
    def __init__(self, kmer_size, fasta_df=None):
        self.k = kmer_size
        self.gene_info_path = "../data/gene_info.json"
        self.pretained_model_path = "./Output/Model/"
        self.weight = False
        with open(gene_info_path, 'r') as json_file:
            self.gene_info = json.load(json_file)
        self.fasta_df=fasta_df
        self.domaine="ACDEFGHIKLMNPQRSTVWYX"

    def load_models(self, model_paths):
        """Load multiple models from given paths."""
        models = []
        for path in model_paths:
            model = load_model(path)
            models.append(model)
        return models

    def predict_with_models(self, models, X_test):
        """Make predictions with each model."""
        predictions = []
        for model in models:
            pred = model.predict(X_test)
            predictions.append(pred)
        return predictions
    
    def weighted_average_predictions(self, predictions, weights):
        """Compute the weighted average of predictions."""
        weighted_preds = np.zeros(predictions[0].shape)
        for pred, weight in zip(predictions, weights):
            weighted_preds += weight * pred
        return weighted_preds / sum(weights)

    def load_fasta_file(self, fasta_path):
        """Read a FASTA file and return a list of sequences."""
        self.fasta_df = fasta.read(fasta_path)

    def build_X_test(self):
        """Convert sequences to k-mer representation for testing."""
        X_test, _, _ = DNA.build_kmer_representation_v2(
            self.fasta_df, domaine=self.domaine, k=3, dtypes=['float64', 'int64'], asCudaDF=False, batch_size=1000, feature_mask=None
        )
        return X_test

    def predict(self, fasta_path):
        """Load models, process FASTA file, and make predictions."""
        X_test = self.build_X_test()
        
        model_paths = [f"{self.pretained_model_path}/{info['file_code']}/FEEDFORWARD_k{self.k}.weights.h5" for gene, info in self.gene_info]
        models = self.load_models(model_paths)
        data_sizes = [self.gene_info[gene]['count']*2 for gene in self.gene_info.keys()]
        weights = [size / sum(data_sizes) for size in data_sizes]
        
        # Make predictions
        predictions = self.predict_with_models(models, X_test)
        
        # Compute final weighted average prediction
        if self.weight:
            predictions = self.weighted_average_predictions(predictions, weights)
        
        # If binary classification, threshold the predictions
        final_prediction_binary = (final_prediction > 0.5).astype(int)
        
        return final_prediction_binary

# Example usage
kmer_size = 3
gene_info_path = 'path_to_gene_info.json'
fasta_path = 'path_to_test.fasta'
single_k_model = SingleKModel(kmer_size, gene_info_path)
predictions = single_k_model.predict(fasta_path)
print(predictions)