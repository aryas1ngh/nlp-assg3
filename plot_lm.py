import json
import matplotlib.pyplot as plt
import os

def plot_lm_metrics(metrics_path='metrics.json', output_dir='plots'):
    if not os.path.exists(metrics_path):
        print(f"Error: {metrics_path} not found.")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    all_steps = []
    all_losses = []
    val_points_x = []
    val_points_y = []

    for epoch_name, data in metrics.items():
        # epoch_num = int(epoch_name.split('_')[1])
        steps_data = data.get('steps', {})
        # Sort steps numerically
        sorted_steps = sorted(steps_data.keys(), key=int)
        
        for step in sorted_steps:
            all_steps.append(int(step))
            all_losses.append(steps_data[step]['train_loss'])
        
        if 'val_loss' in data:
            # Place validation loss at the last step of the epoch
            if sorted_steps:
                val_points_x.append(int(sorted_steps[-1]))
                val_points_y.append(data['val_loss'])

    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(all_steps, all_losses, label='Train Loss (per step)', alpha=0.6)
    if val_points_x:
        plt.scatter(val_points_x, val_points_y, color='red', label='Val Loss (epoch end)', zorder=5)
    
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.title('LM Training Loss Curve (Task 2)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'lm_loss_curve.png'))
    print(f"LM plots updated in {output_dir}/lm_loss_curve.png")

if __name__ == '__main__':
    plot_lm_metrics()
