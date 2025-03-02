import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
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



def plot_hyperparameter_comparison(
    hp_data: Dict[str, Dict[str, Dict[float, List[Union[np.ndarray, pd.DataFrame]]]]],
    hyperparameters: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    window_size: int = 10000,
    figsize: Tuple[int, int] = None,
    save_path: Optional[Path] = None,
    y_limits: Optional[Dict[str, Tuple[float, float]]] = None,
    log_scale: Optional[Dict[str, bool]] = None,
    max_x_value: Optional[int] = 1000000  # Add parameter to control x-axis limit
):
    """
    Create plots comparing different hyperparameter values for selected metrics.
    
    Args:
        hp_data: Nested dictionary with structure {hyperparameter: {metric: {hp_value: [data for each run]}}}
        hyperparameters: List of hyperparameters to plot (if None, plot all)
        metrics: List of metrics to plot (if None, plot all)
        window_size: Size of the window for smoothing
        figsize: Figure size (automatically calculated if None)
        save_path: Path to save the figure
        y_limits: Dictionary of y-axis limits for each metric {metric: (min, max)}
        log_scale: Dictionary specifying whether to use log scale for each metric {metric: bool}
        max_x_value: Maximum value for x-axis (default: 1,000,000)
    
    Returns:
        matplotlib figure
    """
    # Apply ICML 2022 style
    plt.rcParams.update(bundles.icml2022())
    plt.rcParams.update(axes.lines())
    plt.rcParams.update({
        "figure.dpi": 300,
        'font.size': 14,
        'axes.titlesize': 14,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'legend.title_fontsize': 14,
        'text.usetex': True,  # Enable LaTeX rendering
        'text.latex.preamble': r'\usepackage{amsmath}'  # For better math support
    })
    
    # If hyperparameters not specified, use all
    if hyperparameters is None:
        hyperparameters = list(hp_data.keys())
    
    # Determine all metrics if not specified
    all_metrics = set()
    for hp in hyperparameters:
        all_metrics.update(hp_data[hp].keys())
    
    if metrics is None:
        metrics = list(all_metrics)
    
    # Calculate figure dimensions
    n_rows = len(hyperparameters)
    n_cols = len(metrics)
    print(f"n_rows: {n_rows}, n_cols: {n_cols}")
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    # Function to convert hyperparameter names to LaTeX format
    def format_hp_name(hp_name):
        # Convert names like 'beta_0' to '\beta_0'
        greek_letters = {
            'beta': r'\beta',
            'eta': r'\eta',
        }
        
        # Check if hp_name starts with any Greek letter name
        for letter_name, latex_symbol in greek_letters.items():
            if hp_name.startswith(letter_name):
                # Replace the Greek letter name with its LaTeX symbol
                hp_name = hp_name.replace(letter_name, latex_symbol, 1)
                break
                
        # Replace underscores with subscripts
        hp_name = hp_name.replace('_', '_{') + '}'.rjust(hp_name.count('_'))
        
        return hp_name
    
    # Process each hyperparameter and metric
    for row_idx, hp in enumerate(hyperparameters):
        for col_idx, metric in enumerate(metrics):
            ax = axs[row_idx, col_idx]
            
            # Skip if this hyperparameter doesn't have data for this metric
            if metric not in hp_data[hp]:
                ax.text(0.5, 0.5, f"No data for\n{metric}", 
                        ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Get all hyperparameter values for this metric
            hp_values = list(hp_data[hp][metric].keys())
            hp_values.sort()  # Sort values for consistent visualization
            print(f"hp_values for {hp}, {metric}: {hp_values}")
            colors = sns.color_palette("colorblind", len(hp_values))
            markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'H', '+', 'x'][:len(hp_values)]
            
            # Set x-axis limit based on max_x_value
            max_steps = max_x_value
            
            # Format the hyperparameter name for LaTeX
            latex_hp = format_hp_name(hp)
            
            # Plot each hyperparameter value
            for i, hp_value in enumerate(hp_values):
                run_data_list = hp_data[hp][metric][hp_value]
                
                # Create a handle for the legend (we'll only add it once)
                legend_handle = None
                
                for j, run_data in enumerate(run_data_list):
                    # Handle both DataFrame and numpy array formats
                    if isinstance(run_data, pd.DataFrame):
                        # Extract steps and values from DataFrame
                        steps = run_data['Step'].values
                        values = run_data['Value'].values
                    else:
                        # Assume it's a numpy array and create evenly spaced steps
                        values = run_data
                        steps = np.arange(0, len(values) * window_size, window_size)
                    
                    # Ensure steps don't exceed max_x_value
                    mask = steps <= max_steps
                    steps = steps[mask]
                    values = values[mask]
                    
                    # Plot with consistent color but varying alpha for multiple runs
                    line = ax.plot(steps, values, 
                            color=colors[i], alpha=0.8, linewidth=1.5,
                            marker=markers[i], markersize=5, markevery=max(1, len(steps)//20),
                            label=None)  # Don't set label here
                    
                    # Save the line for legend if this is the first run
                    if j == 0:
                        legend_handle = line[0]
                
                # Add a single legend entry for this hyperparameter value with LaTeX formatting
                if legend_handle is not None:
                    # Create LaTeX formatted label
                    latex_label = f"${latex_hp} = {hp_value}$"
                    
                    # Create a custom legend entry
                    ax.plot([], [], color=colors[i], marker=markers[i], markersize=5,
                           label=latex_label, linewidth=1.5)
            
            # Set axis labels and limits
            ax.set_xlabel('Time Steps')
            ax.set_xlim(0, max_steps)
            
            # Format y-axis label
            if "mean" in metric.lower():
                ax.set_ylabel("Cumulative Average Return")
            elif "win" in metric.lower():
                ax.set_ylabel("Cumulative Win Rate")
            else:
                ax.set_ylabel(metric.replace('_', ' ').title())
                
            # Apply log scale if specified
            if log_scale and metric in log_scale and log_scale[metric]:
                print(f"Setting log scale for {metric}")
                ax.set_yscale('log')
            
            # Apply y-limits if specified
            if y_limits and metric in y_limits:
                ax.set_ylim(y_limits[metric])
            
            # Add grid and legend
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Set title to show which hyperparameter and metric with LaTeX formatting
            # latex_metric = metric.replace('_', '\_')  # Escape underscores for LaTeX
            # ax.set_title(f"${latex_hp}$ - {latex_metric}")
            
            # Add a proper legend with no duplicates and LaTeX formatting
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
                
    fig.tight_layout(pad=3.0)
    
    if save_path:
        if not str(save_path).endswith('.pdf'):
            save_path = Path(str(save_path) + '.pdf')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    
    return fig