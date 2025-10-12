import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from flwr.server.strategy import FedProx
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ==================== DATA PREPROCESSING ====================

def load_and_preprocess_data(csv_path):
    """Load and preprocess the student performance dataset"""
    df = pd.read_csv("dataset1_with_output.csv")
    
    # Create output labels from G3 if not present
    if 'output' not in df.columns and 'G3' in df.columns:
        df['output'] = pd.cut(df['G3'], bins=[-1, 6, 15, 20], labels=[0, 1, 2])
    
    # Drop G1, G2, G3 as they shouldn't be used for prediction
    for col in ['G1', 'G2', 'G3']:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Ensure output is numeric
    df['output'] = pd.to_numeric(df['output'], errors='coerce').fillna(0).astype(int)
    
    # Separate features and target
    X = df.drop('output', axis=1)
    y = df['output'].values
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = X.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values.astype(float))
    
    return X_scaled, y, scaler, label_encoders

def partition_data(X, y, num_clients=20):
    """Partition data among clients (non-IID distribution, suitable for FedProx)"""
    num_samples = len(X)
    indices = np.random.permutation(num_samples)
    
    # Split indices among clients
    client_indices = np.array_split(indices, num_clients)
    
    client_data = []
    for idx in client_indices:
        X_client = X[idx]
        y_client = y[idx]
        
        # Simple train-test split without stratification (handles non-IID data)
        if len(y_client) >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X_client, y_client, test_size=0.2, random_state=42
            )
        else:
            # If too few samples, put all in train
            X_train, y_train = X_client, y_client
            X_test, y_test = np.empty((0, X.shape[1])), np.empty((0,), dtype=int)
        
        client_data.append({
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        })
    
    return client_data

# ==================== MODEL DEFINITION ====================

class StudentPerformanceModel(nn.Module):
    """Neural network for student performance prediction"""
    def __init__(self, input_dim, hidden_dim=64, output_dim=3):
        super(StudentPerformanceModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ==================== FLOWER CLIENT ====================

class StudentClient(fl.client.NumPyClient):
    """Flower client for federated learning"""
    def __init__(self, cid, model, trainloader, testloader, device):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Get proximal_mu from config (FedProx parameter)
        proximal_mu = config.get("proximal_mu", 0.01)
        
        # Train
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Save global model parameters for proximal term
        global_params = [p.clone().detach() for p in self.model.parameters()]
        
        epochs = config.get("local_epochs", 5)
        for epoch in range(epochs):
            for batch_X, batch_y in self.trainloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Add proximal term (FedProx)
                proximal_term = 0.0
                for local_param, global_param in zip(self.model.parameters(), global_params):
                    proximal_term += ((local_param - global_param) ** 2).sum()
                loss += (proximal_mu / 2) * proximal_term
                
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in self.testloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss += criterion(outputs, batch_y).item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        loss_avg = loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return float(loss_avg), len(self.testloader.dataset), {"accuracy": float(accuracy)}

# ==================== CLIENT GENERATION ====================

def client_fn(cid: str, client_data_list, input_dim, device):
    """Create a Flower client"""
    client_idx = int(cid)
    client_data = client_data_list[client_idx]
    
    # Create datasets
    X_train = torch.FloatTensor(client_data['X_train'])
    y_train = torch.LongTensor(client_data['y_train'])
    X_test = torch.FloatTensor(client_data['X_test'])
    y_test = torch.LongTensor(client_data['y_test'])
    
    trainset = TensorDataset(X_train, y_train)
    testset = TensorDataset(X_test, y_test)
    
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    testloader = DataLoader(testset, batch_size=16)
    
    # Create model
    model = StudentPerformanceModel(input_dim=input_dim).to(device)
    
    # Convert NumPyClient to Client
    numpy_client = StudentClient(cid, model, trainloader, testloader, device)
    return numpy_client.to_client()

# ==================== PLOTTING ====================

def plot_results(history, num_rounds):
    """Generate clean and relevant plots"""
    
    # Extract accuracies safely
    cent_acc = []
    dist_acc = []
    
    # Try to extract centralized metrics
    if hasattr(history, 'metrics_centralized') and history.metrics_centralized:
        if 'accuracy' in history.metrics_centralized:
            acc_data = history.metrics_centralized['accuracy']
            if acc_data and isinstance(acc_data[0], tuple):
                cent_acc = [acc for _, acc in acc_data]
    
    # Try to extract distributed metrics
    if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
        if 'accuracy' in history.metrics_distributed:
            acc_data = history.metrics_distributed['accuracy']
            if acc_data and isinstance(acc_data[0], tuple):
                dist_acc = [acc for _, acc in acc_data]
    
    # If no data found
    if not cent_acc:
        print("Warning: No accuracy metrics found in history.")
        return
    
    # Create rounds array based on data length
    rounds = np.arange(len(cent_acc))
    
    # Use distributed if available, otherwise use centralized
    if not dist_acc:
        dist_acc = cent_acc
    else:
        # Align lengths
        if len(dist_acc) < len(cent_acc):
            dist_acc = dist_acc + [dist_acc[-1]] * (len(cent_acc) - len(dist_acc))
        elif len(dist_acc) > len(cent_acc):
            dist_acc = dist_acc[:len(cent_acc)]
    
    print(f"\nPlotting {len(cent_acc)} rounds of data")
    print(f"Initial accuracy: {cent_acc[0]:.4f}")
    print(f"Final accuracy: {cent_acc[-1]:.4f}")
    print(f"Best accuracy: {max(cent_acc):.4f} at round {cent_acc.index(max(cent_acc))}")
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FedProx Training Results - Student Performance Prediction', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: Accuracy vs Communication Rounds
    ax1 = axes[0, 0]
    ax1.plot(rounds, cent_acc, 'r-o', label='Global Test Accuracy', linewidth=2.5, markersize=5)
    ax1.plot(rounds, dist_acc, 'b--s', label='Client Avg Accuracy', linewidth=2, markersize=4, alpha=0.7)
    ax1.set_xlabel('Communication Rounds', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Communication Rounds', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Accuracy Improvement per Round
    ax2 = axes[0, 1]
    acc_improvement = [0] + [cent_acc[i] - cent_acc[i-1] for i in range(1, len(cent_acc))]
    colors = ['green' if x >= 0 else 'red' for x in acc_improvement]
    ax2.bar(rounds, acc_improvement, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Communication Rounds', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy Change', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Round Accuracy Improvement', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Convergence Analysis with Moving Average
    ax3 = axes[1, 0]
    window_size = min(5, max(1, len(cent_acc) // 10))
    if len(cent_acc) >= window_size:
        moving_avg = np.convolve(cent_acc, np.ones(window_size)/window_size, mode='valid')
        ax3.plot(rounds, cent_acc, 'b-', alpha=0.3, linewidth=1.5, label='Raw Accuracy')
        ax3.plot(range(window_size-1, window_size-1+len(moving_avg)), moving_avg, 
                'r-', linewidth=3, label=f'{window_size}-Round Moving Avg')
    else:
        ax3.plot(rounds, cent_acc, 'b-', linewidth=2.5, label='Accuracy')
    ax3.set_xlabel('Communication Rounds', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Convergence Analysis', fontsize=13, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Plot 4: Training Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    best_acc = max(cent_acc)
    best_round = cent_acc.index(best_acc)
    final_acc = cent_acc[-1]
    initial_acc = cent_acc[0]
    improvement = final_acc - initial_acc
    avg_last_10 = np.mean(cent_acc[-10:]) if len(cent_acc) >= 10 else np.mean(cent_acc)
    
    summary_text = f"""
    ╔═══════════════════════════════════════╗
    ║   FEDERATED LEARNING SUMMARY (FedProx)   ║
    ╚═══════════════════════════════════════╝
    
    Configuration:
    • Number of Clients: 20
    • Communication Rounds: {len(cent_acc) - 1}
    • Proximal Parameter (μ): 0.01
    • Local Epochs per Round: 5
    
    Performance Metrics:
    • Initial Accuracy: {initial_acc:.4f} ({initial_acc*100:.2f}%)
    • Final Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)
    • Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)
    • Best Round: {best_round}
    • Total Improvement: {improvement:.4f} ({improvement*100:.2f}%)
    
    Convergence:
    • Avg Last 10 Rounds: {avg_last_10:.4f}
    • Std Dev Last 10: {np.std(cent_acc[-10:]):.4f}
    
    Status: {'✓ Converged' if np.std(cent_acc[-5:]) < 0.01 else '◆ Training'}
    """
    
    ax4.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='lightblue', alpha=0.3, edgecolor='navy', linewidth=2))
    
    plt.tight_layout()
    plt.savefig('fedprox_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'fedprox_results.png'")
    plt.show()

# ==================== MAIN EXECUTION ====================

def main():
    # Configuration
    CSV_PATH = 'student-mat.csv'  # Change this to your CSV file path
    NUM_CLIENTS = 20
    NUM_ROUNDS = 50
    PROXIMAL_MU = 0.01  # FedProx hyperparameter
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    X, y, scaler, label_encoders = load_and_preprocess_data(CSV_PATH)
    input_dim = X.shape[1]
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Partition data among clients
    print(f"\nPartitioning data among {NUM_CLIENTS} clients...")
    client_data_list = partition_data(X, y, NUM_CLIENTS)
    
    for i, client in enumerate(client_data_list):
        train_classes = np.bincount(client['y_train'], minlength=3) if len(client['y_train']) > 0 else [0,0,0]
        test_classes = np.bincount(client['y_test'], minlength=3) if len(client['y_test']) > 0 else [0,0,0]
        print(f"Client {i}: Train={len(client['X_train'])} {train_classes}, Test={len(client['X_test'])} {test_classes}")
    
    # Create global test set from all client test sets
    X_test_global = np.vstack([client['X_test'] for client in client_data_list if len(client['X_test']) > 0])
    y_test_global = np.hstack([client['y_test'] for client in client_data_list if len(client['y_test']) > 0])
    print(f"\nGlobal test set: {len(y_test_global)} samples")
    print(f"Global test class distribution: {np.bincount(y_test_global)}")
    
    # Define strategy with FedProx
    def get_evaluate_fn(model, X_test, y_test):
        def evaluate(server_round, parameters, config):
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)
            
            if len(X_test) == 0:
                return 0.0, {"accuracy": 0.0}
            
            model.eval()
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            y_test_tensor = torch.LongTensor(y_test).to(device)
            
            with torch.no_grad():
                outputs = model(X_test_tensor)
                _, predicted = torch.max(outputs.data, 1)
                accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
            
            print(f"Round {server_round} - Global Test Accuracy: {accuracy:.4f}")
            return 0.0, {"accuracy": accuracy}
        
        return evaluate
    
    # Create initial model
    initial_model = StudentPerformanceModel(input_dim=input_dim).to(device)
    initial_parameters = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    
    strategy = FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        proximal_mu=PROXIMAL_MU,
        evaluate_fn=get_evaluate_fn(initial_model, X_test_global, y_test_global),
        on_fit_config_fn=lambda rnd: {"local_epochs": 5, "proximal_mu": PROXIMAL_MU},
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
    )
    
    # Start simulation
    print(f"\n{'='*60}")
    print(f"Starting Federated Learning with FedProx (μ={PROXIMAL_MU})")
    print(f"Number of rounds: {NUM_ROUNDS}")
    print(f"{'='*60}\n")
    
    history = fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(cid, client_data_list, input_dim, device),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    
    # Extract metrics
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    # Plot results
    plot_results(history, NUM_ROUNDS)
    
    return history

if __name__ == "__main__":
    main()