import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import seaborn as sns

def process_and_plot_results(model_dataframes: Dict[str, List[pd.DataFrame]], 
                           window_size: int = 10000,
                           value_column: str = 'Value',
                           yaxis_label: str = 'Cumulative Average Reward',
                           step_column: str = 'Step'):
    """
    Process multiple RL training runs for different models and create a unified visualization.
    
    Args:
        model_dataframes: Dictionary mapping model names to lists of dataframes containing results from different seeds
        window_size: Desired step size for unified x-axis
        value_column: Name of the column containing the reward values
        step_column: Name of the column containing the step values
    """
    # Get a color palette for different models
    colors = sns.color_palette("husl", len(model_dataframes))
    
    plt.figure(figsize=(12, 6))
    
    # Process each model's data
    all_min_steps = []
    all_max_steps = []
    
    # First pass to find global step range
    for model_name, dataframes in model_dataframes.items():
        processed_dfs = []
        
        for df in dataframes:
            # Create bins based on window_size
            df['bin'] = df[step_column] // window_size * window_size
            processed = df.groupby('bin')[value_column].mean().reset_index()
            processed_dfs.append(processed)
        
        min_step = max(df['bin'].min() for df in processed_dfs)
        max_step = min(df['bin'].max() for df in processed_dfs)
        
        all_min_steps.append(min_step)
        all_max_steps.append(max_step)
    
    # Use global min and max steps for unified x-axis
    global_min_step = max(all_min_steps)
    global_max_step = min(all_max_steps)
    unified_steps = np.arange(global_min_step, global_max_step + window_size, window_size)
    
    # Second pass to process and plot each model
    for (model_name, dataframes), color in zip(model_dataframes.items(), colors):
        processed_dfs = []
        
        for df in dataframes:
            df['bin'] = df[step_column] // window_size * window_size
            processed = df.groupby('bin')[value_column].mean().reset_index()
            processed_dfs.append(processed)
        
        # Interpolate values for each run to match unified steps
        interpolated_values = []
        
        for df in processed_dfs:
            df_filtered = df[(df['bin'] >= global_min_step) & (df['bin'] <= global_max_step)]
            interpolated = np.interp(unified_steps, 
                                   df_filtered['bin'], 
                                   df_filtered[value_column])
            interpolated_values.append(interpolated)
        
        values_array = np.array(interpolated_values)
        
        # Calculate statistics
        median_values = np.median(values_array, axis=0)
        min_values = np.min(values_array, axis=0)
        max_values = np.max(values_array, axis=0)
        
        # Plot median line and min-max range for this model
        plt.plot(unified_steps, median_values, label=f'{model_name} (Median)', 
                color=color, linewidth=2)
        plt.fill_between(unified_steps, min_values, max_values,
                        alpha=0.2, color=color) # label=f'{model_name} (Min-Max)'
    
    plt.xlabel('Time Steps')
    plt.ylabel(yaxis_label)
    plt.title('RL Training Results Across Models and Seeds')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

# Example usage:
"""
# Create a dictionary mapping model names to lists of dataframes
model_dfs = {
    'Baseline': [baseline_df1, baseline_df2, baseline_df3],
    'Variant1': [variant1_df1, variant1_df2, variant1_df3],
    'Variant2': [variant2_df1, variant2_df2, variant2_df3]
}

fig = process_and_plot_results(model_dfs, window_size=10000)
plt.show()
"""