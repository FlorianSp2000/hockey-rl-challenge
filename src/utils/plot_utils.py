import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import seaborn as sns
from tueplots import bundles, axes, fontsizes
from pathlib import Path

def process_and_plot_results_multi(data_configs: List[Dict], 
                                 window_size: int = 10000,
                                 figsize=(15, 5),
                                save_path: Path = None):
    """
    Process multiple RL training runs and create subplots with shared legend.
    
    Args:
        data_configs: List of dictionaries, each containing:
            - model_dataframes: Dict[str, List[pd.DataFrame]] - model data
            - value_column: str - column name for values (default: 'Value')
            - yaxis_label: str - y-axis label
            - step_column: str - column name for steps (default: 'Step')
            - title: str - subplot title (optional)
    """
    # Apply ICML 2022 style
    plt.rcParams.update(bundles.icml2022())
    plt.rcParams.update(axes.lines())
    plt.rcParams.update({
        "figure.dpi": 300,
        'font.size': 14,  # Base font size
        'axes.titlesize': 16,  # Subplot titles
        'axes.labelsize': 16,  # Axis labels
        'xtick.labelsize': 14,  # X-axis tick labels
        'ytick.labelsize': 14,  # Y-axis tick labels
        'legend.fontsize': 12,  # Legend text
        'legend.title_fontsize': 14  # Legend title
    })

    n_plots = len(data_configs)
    fig, axs = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axs = [axs]

    # Get a consistent color palette for all subplots
    all_models = set()
    for config in data_configs:
        all_models.update(config.get('model_dataframes', {}).keys())
    colors = dict(zip(all_models, sns.color_palette("colorblind", len(all_models))))
    markers = dict(zip(all_models, ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'H', '+', 'x'][:len(all_models)]))

    with plt.rc_context(bundles.icml2022()):
        # Process each subplot
        for ax_idx, (ax, config) in enumerate(zip(axs, data_configs)):
            model_dataframes = config.get('model_dataframes', {})
            value_column = config.get('value_column', 'Value')
            step_column = config.get('step_column', 'Step')
            yaxis_label = config.get('yaxis_label', '')
            title = config.get('title', '')
            use_log_scale = config.get('use_log_scale', False)
            y_limits = config.get('y_limits', None)

            # Find global step range for this subplot
            all_min_steps = []
            all_max_steps = []
            
            for model_name, dataframes in model_dataframes.items():
                processed_dfs = []
                for df in dataframes:
                    df['bin'] = df[step_column] // window_size * window_size
                    processed = df.groupby('bin')[value_column].mean().reset_index()
                    processed_dfs.append(processed)
                
                min_step = max(df['bin'].min() for df in processed_dfs)
                max_step = min(df['bin'].max() for df in processed_dfs)
                all_min_steps.append(min_step)
                all_max_steps.append(max_step)
            
            global_min_step = max(all_min_steps)
            global_max_step = min(all_max_steps)
            unified_steps = np.arange(global_min_step, global_max_step + window_size, window_size)

            # Plot each model
            for model_name, dataframes in model_dataframes.items():
                processed_dfs = []
                for df in dataframes:
                    df['bin'] = df[step_column] // window_size * window_size
                    processed = df.groupby('bin')[value_column].mean().reset_index()
                    processed_dfs.append(processed)
                
                interpolated_values = []
                for df in processed_dfs:
                    df_filtered = df[(df['bin'] >= global_min_step) & (df['bin'] <= global_max_step)]
                    interpolated = np.interp(unified_steps, 
                                           df_filtered['bin'], 
                                           df_filtered[value_column])
                    interpolated_values.append(interpolated)
                
                values_array = np.array(interpolated_values)
                mean_values = np.mean(values_array, axis=0)
                min_values = np.min(values_array, axis=0)
                max_values = np.max(values_array, axis=0)
                
                ax.plot(unified_steps, mean_values, 
                       label=model_name if ax_idx == 0 else "", # Only label in first subplot
                       color=colors[model_name], 
                       linewidth=2, 
                       marker=markers[model_name], 
                       markevery=len(unified_steps)//10,
                       markersize=6)
                ax.fill_between(unified_steps, min_values, max_values,
                              alpha=0.2, color=colors[model_name])
                if use_log_scale:
                    ax.set_yscale('log')
                    ax.set_ylim(bottom=max(1e-5, min_values.min()), top=max_values.max()*1.1)
                if y_limits:
                    ax.set_ylim(y_limits)

            ax.set_xlabel('Time Steps')
            ax.set_ylabel(yaxis_label)
            if title:
                ax.set_title(title)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            if max(unified_steps) > 1e6:
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        # Add shared legend below the subplots
        fig.legend(
            loc='center',
            bbox_to_anchor=(0.5, -0.1),
            ncol=len(all_models),
            # title='Methods',
            fontsize=14,  # Increased text size
            handlelength=2,  # Increase length of lines in legend
            markerscale=2.0,  # Make markers bigger in legend

        )

        # Adjust layout
        fig.tight_layout(pad=3.0)
        plt.subplots_adjust(bottom=0.05)  # Make room for legend
        if save_path:
            if not save_path.suffix == '.pdf':
                save_path = save_path.with_suffix('.pdf')
            plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        
    return fig

