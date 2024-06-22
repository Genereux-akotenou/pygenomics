import pandas as pd
import ipywidgets as widgets
from IPython.display import display, FileLink
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class GenBoard:
    def __init__(self, dataframe: pd.DataFrame):
        self.prediction = dataframe.sort_index(axis=1)
        self.thresholds = [0.25, 0.5, 0.75, 0.95]
        self.init_df = None
        self.kmer_df = None
        self.threshold_slider = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.05,
            description='Threshold:',
            continuous_update=False
        )
        self.tab = widgets.Tab()

    def get(self):
        return self.prediction

    def add_initial_set(self, df):
        self.init_df = df

    def add_kmer_set(self, df):
        self.kmer_df = df
        
    def classification_report(self):
        threshold = self.threshold_slider.value
        binary_prediction = (self.prediction > threshold).astype(int)
        return binary_prediction

    def create_report_tab(self):
        report_output = widgets.Output()
    
        def update_report(change):
            with report_output:
                report_output.clear_output(wait=True)
                threshold = change['new']
                binary_prediction = (self.prediction > threshold).astype(int)
                
                plt.figure(figsize=(10, 6))
                gene_counts = binary_prediction.sum(axis=0)
                genes = self.prediction.columns
                
                plt.bar(genes, gene_counts, align='center')
                plt.title('Number of Predicted Genes per Class')
                plt.xlabel('Gene Family')
                plt.ylabel('Number of Predicted Genes')
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.show()
    
        self.threshold_slider.observe(update_report, names='value')
    
        with report_output:
            update_report({'new': self.threshold_slider.value})
    
        return widgets.VBox([self.threshold_slider, report_output])


    def create_data_transformation_tab(self):
        init_df_output = widgets.Output()
        kmer_df_output = widgets.Output()
        
        def save_df(df, filename):
            df.to_csv(filename)
            return FileLink(filename)

        with init_df_output:
            display(self.init_df)
        
        with kmer_df_output:
            display(self.kmer_df)

        init_df_button = widgets.Button(description="Download Initial DataFrame")
        kmer_df_button = widgets.Button(description="Download Transformed DataFrame")

        init_df_file_link = widgets.HTML()
        kmer_df_file_link = widgets.HTML()

        def on_init_df_button_clicked(b):
            filename = 'initial_dataframe.csv'
            save_df(self.init_df, filename)
            init_df_file_link.value = f'<a href="{filename}" download>Download Initial DataFrame</a>'

        def on_kmer_df_button_clicked(b):
            filename = 'transformed_dataframe.csv'
            save_df(self.kmer_df, filename)
            kmer_df_file_link.value = f'<a href="{filename}" download>Download Transformed DataFrame</a>'

        init_df_button.on_click(on_init_df_button_clicked)
        kmer_df_button.on_click(on_kmer_df_button_clicked)

        init_df_vbox = widgets.VBox([widgets.HTML('<h3>Initial DataFrame</h3>'), init_df_output, init_df_button, init_df_file_link])
        kmer_df_vbox = widgets.VBox([widgets.HTML('<h3>Transformed DataFrame</h3>'), kmer_df_output, kmer_df_button, kmer_df_file_link])
        
        return widgets.VBox([init_df_vbox, kmer_df_vbox])

    def create_empty_tab(self):
        return widgets.VBox([widgets.HTML('<h3>Model Pipeline</h3>'), widgets.HTML('<p>To be implemented</p>')])

    def display(self):
        self.tab.children = [self.create_report_tab(), self.create_data_transformation_tab(), self.create_empty_tab()]
        self.tab.set_title(0, 'Report')
        self.tab.set_title(1, 'Data Transformation')
        self.tab.set_title(2, 'Model Pipeline')
        display(self.tab)