import pandas as pd
import ipywidgets as widgets
from IPython.display import display, FileLink
from matplotlib.patches import Rectangle, FancyBboxPatch, Arrow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import platform

class GenBoard:
    def __init__(self, dataframe: pd.DataFrame):
        self.prediction = dataframe.sort_index(axis=1)
        self.init_df = None
        self.kmer_df = None
        self.model_dict = {}
        self.threshold_slider = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.05,
            description='Threshold:',
            continuous_update=False
        )
        self.voting_method = widgets.RadioButtons(
            options=['Max Voting', 'Weighted Max Voting', 'Two-Stage Voting'],
            description='Voting Method:',
            disabled=False
        )
        self.model_weights = np.ones(self.prediction.shape[1])
        self.tab = widgets.Tab(layout=widgets.Layout(minwidth='1200px', height='750px'))
        self.two_stage_prediction = None

    def get(self):
        return self.prediction

    def add_initial_set(self, df):
        self.init_df = df

    def add_kmer_set(self, df):
        self.kmer_df = df

    def add_model(self, model_dict):
        self.model_dict = model_dict

    def add_stage2_result(self, df):
        self.two_stage_prediction = df

    def create_report_tab(self):
        report_output = widgets.Output()
    
        def update_report(change):
            with report_output:
                report_output.clear_output(wait=True)
                threshold = self.threshold_slider.value
                voting_method = self.voting_method.value
    
                if voting_method == 'Max Voting':
                    binary_prediction = (self.prediction > threshold).astype(int)
                    final_prediction = binary_prediction.idxmax(axis=1)
                    all_below_threshold = (self.prediction.max(axis=1) <= threshold)
                    final_prediction[all_below_threshold] = 'Unknown'
                elif voting_method == 'Weighted Max Voting':
                    weighted_prediction = self.prediction * self.model_weights
                    binary_prediction = (weighted_prediction > threshold).astype(int)
                    final_prediction = binary_prediction.idxmax(axis=1)
                    all_below_threshold = (weighted_prediction.max(axis=1) <= threshold)
                    final_prediction[all_below_threshold] = 'Unknown'
                elif voting_method == 'Two-Stage Voting':
                    if self.two_stage_prediction is not None:
                        binary_prediction = (self.two_stage_prediction > threshold).astype(int)
                        final_prediction = binary_prediction.idxmax(axis=1)
                        all_below_threshold = (self.two_stage_prediction.max(axis=1) <= threshold)
                        final_prediction[all_below_threshold] = 'Unknown'
                    else:
                        raise ValueError("Two-stage prediction data is not available.")
                else:
                    raise ValueError("Unsupported voting method")
    
                gene_counts = final_prediction.value_counts()
                genes = gene_counts.index
    
                # Compute counts of predictions greater than the threshold
                above_threshold_counts = (self.prediction > threshold).sum(axis=0).reindex(genes, fill_value=0)
    
                plt.figure(figsize=(10, 6))
    
                above_threshold_bars = plt.bar(genes, above_threshold_counts, align='center', color='orange', alpha=0.6)
                bars = plt.bar(genes, gene_counts, align='center', color=['#1f77b4' if gene != 'Unknown' else '#eee' for gene in genes])
    
                plt.title('Number of Predicted Genes per Class')
                plt.xlabel('Gene Family')
                plt.ylabel('Number of Predicted Genes')
                plt.xticks(rotation=90)
                plt.tight_layout()
    
                # Add counts on top of bars
                for bar, count in zip(bars, gene_counts):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2.0, height, str(count), ha='center', va='bottom')
                
                # Add counts on top of above_threshold_bars
                for bar, count in zip(above_threshold_bars, above_threshold_counts):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2.0, height, str(count), ha='center', va='bottom')
    
                # Add legend
                plt.legend(['Above Threshold Observations', 'Predicted Genes after voting'], loc='upper right')
    
                plt.show()
    
        self.threshold_slider.observe(update_report, names='value')
        self.voting_method.observe(update_report, names='value')
        with report_output:
            update_report({'new': self.threshold_slider.value})
        controls_layout = widgets.Layout(display='flex', justify_content='space-between', width='100%')
        controls = widgets.HBox([self.threshold_slider, self.voting_method], layout=controls_layout)
        combined_widget = widgets.VBox([controls, report_output])
        return combined_widget


    def create_meta_probabilities_tab(self):
        if 'meta' in self.init_df.columns:
            df  = self.init_df[['meta']].copy()
            df2 = self.init_df[['meta']].copy()
        else:
            df  = self.init_df[['id']].copy()
            df.rename(columns={'id': 'meta'}, inplace=True)
            df2  = self.init_df[['id']].copy()
            df2.rename(columns={'id': 'meta'}, inplace=True)
    
        # Add columns for gene family probabilities from prediction
        for gene_family in self.prediction.columns:
            df[gene_family] = self.prediction[gene_family]
        for gene_family in self.two_stage_prediction.columns:
            df2[gene_family] = self.two_stage_prediction[gene_family]
    
        # Add a prediction column with formatted string
        df['prediction'] = self.prediction.idxmax(axis=1) + ' (' + self.prediction.max(axis=1).astype(str) + ')'
        df2['prediction'] = self.two_stage_prediction.idxmax(axis=1) + ' (' + self.two_stage_prediction.max(axis=1).astype(str) + ')'
    
        # Add a column for unknown gene predictions
        df['Unknown Gene Family'] = '✓'
        df2['Unknown Gene Family'] = '✓'
    
        # Create widgets for threshold, voting method, search bar, and result count
        threshold_slider = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1,
            step=0.01,
            description='Threshold:',
            continuous_update=False
        )
    
        voting_method = widgets.RadioButtons(
            options=['Max Voting', 'Weighted Max Voting', 'Two-Stage Voting'],
            description='Voting Method:',
            disabled=False
        )
        search_bar = widgets.Text(
            value='',
            placeholder='Search meta...',
            description='Filter:',
            continuous_update=True
        )
        result_count = widgets.Label(value="")
    
        df_output = widgets.Output()
    
        def update_df(change):
            with df_output:
                df_output.clear_output(wait=True)
                threshold = threshold_slider.value
                voting = voting_method.value
                search_value = search_bar.value.lower()
    
                # Update the 'Unknown Gene Family' column based on the threshold
                if voting == 'Two-Stage Voting' and self.two_stage_prediction is not None:
                    df2['Unknown Gene Family'] = np.where(self.two_stage_prediction.max(axis=1) < threshold, '✗', '✓')
                else:
                    df['Unknown Gene Family'] = np.where(self.prediction.max(axis=1) < threshold, '✗', '✓')
    
                # Filter DataFrame based on search bar input
                filtered_df = df[df['meta'].str.lower().str.contains(search_value)]
                filtered_df2 = df2[df2['meta'].str.lower().str.contains(search_value)]
    
                # Update result count
                if voting == 'Two-Stage Voting' and self.two_stage_prediction is not None:
                    result_count.value = f"({len(filtered_df2)} matches)"
                else:
                    result_count.value = f"({len(filtered_df)} matches)"
    
                # Highlight cells based on threshold
                def highlight_cells(val):
                    color = 'background-color: #ffcccc' if val < threshold else 'background-color: #a7c942'
                    return color
    
                # Highlight cells in the "Unknown Gene Family" column
                def highlight_unknown_cells(val):
                    return 'background-color: #add8e6' if val == '✗' else ''
    
                if voting == 'Two-Stage Voting' and self.two_stage_prediction is not None:
                    styled_df = filtered_df2.style.applymap(highlight_cells, subset=pd.IndexSlice[:, self.two_stage_prediction.columns])
                else:
                    styled_df = filtered_df.style.applymap(highlight_cells, subset=pd.IndexSlice[:, self.prediction.columns])
                styled_df = styled_df.applymap(highlight_unknown_cells, subset=['Unknown Gene Family'])
                styled_df = styled_df.set_table_styles([{'selector': 'table', 'props': [('min-width', '900px')]}])
                display(styled_df)
    
        # Observe changes in the threshold slider, voting method, and search bar
        threshold_slider.observe(update_df, names='value')
        voting_method.observe(update_df, names='value')
        search_bar.observe(update_df, names='value')
    
        # Initialize the display
        update_df(None)
    
        # Create the layout for the tab
        controls_layout = widgets.Layout(display='flex', justify_content='space-between', width='100%')
        controls = widgets.HBox([threshold_slider, voting_method], layout=controls_layout)
        search_layout = widgets.HBox([search_bar, result_count], layout=widgets.Layout(display='flex', align_items='center'))
        tab_content = widgets.VBox([controls, search_layout, df_output])
        return tab_content


    def create_data_transformation_tab(self):
        init_df_output = widgets.Output()
        kmer_df_output = widgets.Output()

        def save_df(df, filename):
            df.to_csv(filename, index=False)
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

    def create_pipeline_diagram(self):
        # Create a widget to hold the output
        out = widgets.Output()

        with out:
            fig, ax = plt.subplots(figsize=(14, 10))

            # Define component positions
            positions = {
                "load_data": (0.1, 0.8),
                "build_kmer_set": (0.1, 0.6),
                "load_models": (0.4, 0.8),
                "make_predictions": (0.4, 0.6),
                "aggregate_predictions": (0.7, 0.6),
                "generate_report": (0.7, 0.4),
                "display_results": (0.4, 0.4),
            }
            
            # Function to add rectangles for each component
            def add_component(ax, text, position):
                text_width = len(text) * 0.015
                text_height = 0.08
                rect = Rectangle(position, text_width, text_height, linewidth=1, edgecolor='#eee', facecolor='#1f77b4', alpha=0.7)
                ax.add_patch(rect)
                ax.text(position[0] + text_width / 2, position[1] + text_height / 2, text, ha="center", va="center",
                        fontsize=10, color="white", weight="bold")

            components = {
                "Load Data": positions["load_data"],
                "Build K-mer Set": positions["build_kmer_set"],
                "Load Models": positions["load_models"],
                "Make Predictions": positions["make_predictions"],
                "Aggregate Predictions": positions["aggregate_predictions"],
                "Generate Report": positions["generate_report"],
                "Display Results": positions["display_results"],
            }

            for text, position in components.items():
                add_component(ax, text, position)

            # Function to add arrows between components
            def add_arrow(ax, start, end):
                arrow = Arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                              width=0.01, edgecolor="black", facecolor="black", alpha=0.7)
                ax.add_patch(arrow)

            # Define the connections between components using positions
            connections = [
                ("load_data", "build_kmer_set"),
                ("build_kmer_set", "make_predictions"),
                ("load_models", "make_predictions"),
                ("make_predictions", "aggregate_predictions"),
                ("aggregate_predictions", "generate_report"),
                ("generate_report", "display_results"),
            ]

            for start, end in connections:
                add_arrow(ax, positions[start], positions[end])

            # Add titles and descriptions
            ax.text(0.5, 0.95, "Tool Pipeline Diagram", ha="center", va="center", fontsize=16, weight="bold")
            ax.text(0.5, 0.9, "This diagram illustrates the steps, model architecture, and pipeline.", ha="center", va="center", fontsize=12)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

            # Instead of plt.show(), display the figure directly
            plt.close(fig)
            display(fig)

            # Display model summaries for each model used in predictions
            for gene_family in self.prediction.columns:
                if gene_family in self.model_dict:
                    model, features = self.model_dict[gene_family]
                    
                    model_summary = []
                    model.summary(print_fn=lambda x: model_summary.append(x))
                    model_summary = "\n".join(model_summary)
                    
                    display(widgets.HTML(f"<h3>{gene_family} Model Summary</h3>"))
                    display(widgets.HTML(f"<pre>{model_summary}</pre>"))

        return out

    def create_about_tab(self):
        version_info = {
            "Application Name": "GenBoard",
            "Version": "1.0.0",
            "Python Version": platform.python_version(),
            "Operating System": platform.system(),
        }
        version_html = "<h3>About</h3>"
        version_html += "<ul>"
        for key, value in version_info.items():
            version_html += f"<li><b>{key}:</b> {value}</li>"
        version_html += "</ul>"
        return widgets.VBox([widgets.HTML(version_html)])

    
    def display(self):
        self.tab.children = [
            self.create_report_tab(), 
            self.create_meta_probabilities_tab(), 
            self.create_data_transformation_tab(), 
            self.create_pipeline_diagram(), 
            self.create_about_tab()
        ]
        self.tab.set_title(0, 'Stats Report')
        self.tab.set_title(1, 'Meta Report')
        self.tab.set_title(2, 'Data Transformation')
        self.tab.set_title(3, 'Model Pipeline')
        self.tab.set_title(4, 'About')
        display(self.tab)