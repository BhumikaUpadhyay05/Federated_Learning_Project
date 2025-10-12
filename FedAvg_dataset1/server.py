import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
import os

# Aggregate evaluation metrics (accuracy) across clients
def aggregate_metrics(metrics):
    """
    Aggregate accuracy metrics from multiple clients using weighted average.
    
    Args:
        metrics: List of tuples (num_examples, metrics_dict)
    
    Returns:
        Dict with aggregated accuracy
    """
    if not metrics:
        return {}
    
    total_examples = 0
    weighted_accuracy = 0.0
    
    # metrics is a list of tuples: [(num_examples, {"accuracy": value}), ...]
    for num_examples, metric_dict in metrics:
        if "accuracy" in metric_dict:
            weighted_accuracy += num_examples * metric_dict["accuracy"]
            total_examples += num_examples
    
    if total_examples == 0:
        return {}
    
    return {"accuracy": weighted_accuracy / total_examples}

# Define FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    on_fit_config_fn=lambda rnd: {"local_epochs": 5},
    evaluate_metrics_aggregation_fn=aggregate_metrics
)

# Start server
history = fl.server.start_server(
    server_address="127.0.0.1:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=50)
)


# Extract Results
# ============================================
rounds = []
losses = []
accuracies = []

# Extract data from history
for round_num, loss_value in history.losses_distributed:
    rounds.append(round_num)
    losses.append(loss_value)
    
    # Find corresponding accuracy
    accuracy = None
    if "accuracy" in history.metrics_distributed:
        for rnd, acc in history.metrics_distributed["accuracy"]:
            if rnd == round_num:
                accuracy = acc
                break
    accuracies.append(accuracy if accuracy is not None else 0.0)

# Save history to JSON
history_data = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "rounds": rounds,
    "losses": losses,
    "accuracies": accuracies
}

os.makedirs('results', exist_ok=True)
with open('results/training_history.json', 'w') as f:
    json.dump(history_data, f, indent=2)
print("\n✓ Training history saved to 'results/training_history.json'")

# Print text summary
print("\n=== Training Summary ===")
for round_num, loss_value, accuracy in zip(rounds, losses, accuracies):
    if accuracy is not None and accuracy > 0:
        print(f"Round {round_num}: Loss = {loss_value:.4f}, Accuracy = {accuracy:.4f}")
    else:
        print(f"Round {round_num}: Loss = {loss_value:.4f}")



# Visualization Function
# ============================================
def create_visualizations(rounds, losses, accuracies):
    """Create clean, professional visualizations of training results."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E']
    
    # Create figure with better layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Main Training Curves (Loss & Accuracy together)
    # Loss curve
    line1 = ax1.plot(rounds, losses, marker='o', linewidth=2.5, markersize=6,
                    color=colors[0], label='Training Loss', alpha=0.8)
    ax1.set_xlabel('Training Round', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold', color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve on twin axis
    ax1_twin = ax1.twinx()
    line2 = ax1_twin.plot(rounds, [acc * 100 for acc in accuracies], marker='s', 
                         linewidth=2.5, markersize=6, color=colors[1], 
                         label='Accuracy (%)', alpha=0.8)
    ax1_twin.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold', color=colors[1])
    ax1_twin.tick_params(axis='y', labelcolor=colors[1])
    ax1_twin.set_ylim(0, 100)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=10)
    ax1.set_title('Training Progress: Loss & Accuracy', fontsize=13, fontweight='bold', pad=10)
    
    # 2. Accuracy Trend with Confidence
    ax2.plot(rounds, [acc * 100 for acc in accuracies], marker='s', linewidth=2.5,
            color=colors[1], label='Accuracy')
    
    # Add smoothing line for trend
    if len(accuracies) > 3:
        from scipy.ndimage import gaussian_filter1d
        smoothed_acc = gaussian_filter1d(accuracies, sigma=1) * 100
        ax2.plot(rounds, smoothed_acc, '--', linewidth=2, color=colors[3],
                alpha=0.7, label='Trend')
    
    # Highlight best accuracy
    best_idx = np.argmax(accuracies)
    ax2.plot(rounds[best_idx], accuracies[best_idx] * 100, 'o', 
            markersize=10, color='gold', markeredgecolor='black',
            label=f'Best: {accuracies[best_idx]*100:.1f}%')
    
    ax2.set_xlabel('Training Round', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_title('Model Accuracy Progression', fontsize=13, fontweight='bold', pad=10)
    
    # 3. Loss Reduction Progress
    initial_loss = losses[0]
    loss_reduction_pct = [(initial_loss - loss) / initial_loss * 100 for loss in losses]
    
    ax3.fill_between(rounds, loss_reduction_pct, alpha=0.3, color=colors[2])
    ax3.plot(rounds, loss_reduction_pct, linewidth=2.5, color=colors[2], 
            label='Loss Reduction')
    
    # Final reduction value
    final_reduction = loss_reduction_pct[-1]
    ax3.axhline(y=final_reduction, color='red', linestyle='--', alpha=0.7,
               label=f'Final: {final_reduction:.1f}%')
    
    ax3.set_xlabel('Training Round', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Loss Reduction (%)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_title('Cumulative Loss Reduction', fontsize=13, fontweight='bold', pad=10)
    
    # 4. Key Metrics Summary (Clean table)
    ax4.axis('off')
    
    # Calculate key metrics
    best_round = rounds[best_idx]
    peak_accuracy = max(accuracies) * 100
    final_accuracy = accuracies[-1] * 100
    accuracy_gain = (accuracies[-1] - accuracies[0]) * 100
    avg_accuracy = np.mean(accuracies) * 100
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Rounds', f'{len(rounds)}'],
        ['Final Accuracy', f'{final_accuracy:.2f}%'],
        ['Peak Accuracy', f'{peak_accuracy:.2f}%'],
        ['Accuracy Gain', f'+{accuracy_gain:.2f}%'],
        ['Best Round', f'{best_round}'],
        ['Final Loss', f'{losses[-1]:.4f}'],
        ['Loss Reduction', f'{final_reduction:.1f}%'],
        ['Avg Accuracy', f'{avg_accuracy:.2f}%']
    ]
    
    # Create table
    table = ax4.table(cellText=summary_data, cellLoc='center', 
                     loc='center', bbox=[0.1, 0.2, 0.8, 0.6])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    
    # Header styling
    for i in range(2):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white', size=12)
    
    # Highlight important rows
    highlight_rows = [1, 2, 3, 6]  # Rows to highlight
    for row in highlight_rows:
        for col in range(2):
            table[(row, col)].set_facecolor('#f0f8ff')
            table[(row, col)].set_text_props(weight='bold')
    
    ax4.set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)
    
    # Overall title and layout
    strategy_name = "FedAvg"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.suptitle(f'Federated Learning Results - {strategy_name}\n{timestamp}', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Save with high quality
    os.makedirs('results', exist_ok=True)
    filename = f'results/fl_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    
  
    
    # summary
    print("\n" + "="*50)
    print("QUICK SUMMARY")
    print("="*50)
    print(f" Peak Accuracy: {peak_accuracy:.2f}% (Round {best_round})")
    print(f" Final Accuracy: {final_accuracy:.2f}%")
    print(f" Accuracy Gain: +{accuracy_gain:.2f}%")
    print(f" Loss Reduction: {final_reduction:.1f}%")
    print("="*50)

# Generate the visualizations
if len(rounds) > 0 and len(losses) > 0:
    print("\n Generating improved visualizations...")
    try:
        create_visualizations(rounds, losses, accuracies)
    except Exception as e:
        print(f" Visualization error: {e}")
        # Fallback to simple plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(rounds, losses, 'b-o')
        plt.title('Training Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(rounds, [a*100 for a in accuracies], 'r-s')
        plt.title('Accuracy')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
else:
    print(" No data to visualize")