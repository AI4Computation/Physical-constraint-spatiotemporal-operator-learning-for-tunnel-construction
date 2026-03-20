import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import random

class BranchNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BranchNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.PReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)  # 杈撳嚭娼滃湪鐗瑰緛
        )
    
    def forward(self, x):
        return self.net(x)

class TrunkNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TrunkNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 2*hidden_dim),
            nn.PReLU(),
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)  # 杈撳嚭娼滃湪鐗瑰緛
        )
    
    def forward(self, x):
        return self.net(x)

from torch.nn.utils.rnn import pad_sequence

class StressDataset(Dataset):
    def __init__(self, branch_inputs, trunk_inputs, outputs):
        """
        branch_inputs: (N_samples, n_branch_features) - 瀛旀礊鍙傛暟鍜岃竟鐣屾潯浠?
        trunk_inputs: list of (n_points, 2) - 姣忎釜鏍锋湰鐨勭┖闂村潗鏍囩偣 (x, y)
        outputs: list of (n_points,) - 姣忎釜鏍锋湰鐨勫簲鍔涘€?
        """
        self.branch_inputs = torch.tensor(branch_inputs, dtype=torch.float32)
        self.trunk_inputs = [torch.tensor(x, dtype=torch.float32) for x in trunk_inputs]
        self.outputs = [torch.tensor(x, dtype=torch.float32) for x in outputs]
        
        self.lengths = [len(x) for x in outputs]
    
    def __len__(self):
        return len(self.branch_inputs)
    
    def __getitem__(self, idx):
        return (self.branch_inputs[idx], 
                self.trunk_inputs[idx], 
                self.outputs[idx],
                self.lengths[idx])

def collate_fn(batch):
    """
    鑷畾涔夌殑collate鍑芥暟锛岀敤浜庡鐞嗕笉鍚岄暱搴︾殑鏁版嵁
    """
    branch_inputs, trunk_inputs, outputs, lengths = zip(*batch)
    
    branch_inputs = torch.stack(branch_inputs)
    
    trunk_inputs = pad_sequence(trunk_inputs, batch_first=True)
    outputs = pad_sequence(outputs, batch_first=True)
    
    lengths = torch.tensor(lengths)
    
    return branch_inputs, trunk_inputs, outputs, lengths

class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim, output_dim):
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet(branch_input_dim, hidden_dim, output_dim)
        self.trunk_net = TrunkNet(trunk_input_dim, hidden_dim, output_dim)
    
    def forward(self, branch_input, trunk_input, lengths):
        """
        branch_input: (batch_size, n_branch_features)
        trunk_input: (batch_size, max_points, trunk_input_dim)
        lengths: (batch_size,) 姣忎釜鏍锋湰瀹為檯鐨勭偣鏁?
        """
        branch_output = self.branch_net(branch_input)  # (batch_size, output_dim)
        
        batch_size, max_points, _ = trunk_input.shape
        trunk_input = trunk_input.view(-1, trunk_input.shape[-1])
        trunk_output = self.trunk_net(trunk_input)
        trunk_output = trunk_output.view(batch_size, max_points, -1)
        
        output = torch.bmm(trunk_output, branch_output.unsqueeze(2)).squeeze(2)
        
        mask = torch.arange(max_points).expand(batch_size, max_points).to(output.device)
        mask = mask < lengths.unsqueeze(1)
        output = output * mask.float()
        
        return output

import os
import pandas as pd
import re

def extract_params_from_filename(filename):
    """浠庢枃浠跺悕涓彁鍙栧弬鏁板€?""
    base_name = os.path.splitext(filename)[0]#鍘婚櫎鎺夋枃浠舵墿灞曞悕
    params = {}
    parts = base_name.split(';')

    for part in parts:
        key,value = part.split('=')
        if key != 'No':  # 鎺掗櫎鏃犳剰涔夌殑宸ュ喌缂栧彿
            params[key] = float(value) # 杞崲涓烘诞鐐规暟锛堝鏈夊繀瑕侊級    
    param_order = ['k', 'ratio', 'r', 'phi', 'theta', 'a', 'b']
    return np.array([params[key] for key in param_order])

def load_data(folder_path):
    """鍔犺浇鎵€鏈夋暟鎹?""
    branch_inputs = []
    trunk_inputs = []
    outputs = []
    
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    n_samples = len(csv_files)
    
    for filename in csv_files:
        branch_input = extract_params_from_filename(filename)
        branch_inputs.append(branch_input)
        
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path, encoding='gbk')  # 浣跨敤 'gbk' 缂栫爜璇诲彇鏂囦欢
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='utf-8-sig')  # 浣跨敤 'utf-8-sig' 缂栫爜澶勭悊甯?BOM 鐨勬枃浠?
        
        df.columns = [col.strip() for col in df.columns]
        stress_col = [col for col in df.columns if 'S11' in col][0]  # 鎵惧埌鍖呭惈'S11'鐨勫垪鍚?
        
        mask = (df['X'] > -1.6) & (df['X'] < 1.6) & (df['Y'] > -1.6) & (df['Y'] < 1.6)
        filtered_df = df[mask]
        
        points = filtered_df[['X', 'Y']].values
        stress = filtered_df[stress_col].values
        
        trunk_inputs.append(points)
        outputs.append(stress)
    
    return np.array(branch_inputs), trunk_inputs, outputs, n_samples

def polar_coordinates(coords):
    polar_coords = []
    for i in range(len(coords)):
        x_col = coords[i][:, 0]
        y_col = coords[i][:, 1]
        
        rho_col = np.sqrt(x_col**2 + y_col**2) # 璁＄畻鏋佸緞
        theta_col = np.arctan2(y_col, x_col)  # 璁＄畻鏋佽,鍗曚綅涓哄姬搴︼紝theta = np.arctan2(ylist, xlist)
        
        polar_coords.append(np.column_stack((rho_col, theta_col)))
    return polar_coords

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=100):
    train_losses = []
    val_losses = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
    
    def compute_validation_loss(loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for branch_input, trunk_input, target_output, lengths in loader:
                branch_input = branch_input.to(device)
                trunk_input = trunk_input.to(device)
                target_output = target_output.to(device)
                lengths = lengths.to(device)
                
                prediction = model(branch_input, trunk_input, lengths)
                mask = torch.arange(prediction.size(1)).expand(prediction.size(0), -1).to(device)
                mask = mask < lengths.unsqueeze(1)
                loss = criterion(prediction * mask.float(), target_output * mask.float())
                total_loss += loss.item()
        
        return total_loss / len(loader)

    last_lr = optimizer.param_groups[0]['lr']

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for branch_input, trunk_input, target_output, lengths in train_loader:
            branch_input = branch_input.to(device)
            trunk_input = trunk_input.to(device)
            target_output = target_output.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            
            prediction = model(branch_input, trunk_input, lengths)
            
            mask = torch.arange(prediction.size(1)).expand(prediction.size(0), -1).to(device)
            mask = mask < lengths.unsqueeze(1)
            loss = criterion(prediction * mask.float(), target_output * mask.float())
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = compute_validation_loss(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        
        if current_lr != optimizer.param_groups[0]['lr']:
            print(f"Learning rate decreased from {current_lr} to {optimizer.param_groups[0]['lr']}")
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss History')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for branch_input, trunk_input, target_output, lengths in test_loader:
            branch_input = branch_input.to(device)
            trunk_input = trunk_input.to(device)
            target_output = target_output.to(device)
            lengths = lengths.to(device)
            
            prediction = model(branch_input, trunk_input, lengths)
            mask = torch.arange(prediction.size(1)).expand(prediction.size(0), -1).to(device)
            mask = mask < lengths.unsqueeze(1)
            
            masked_pred = prediction * mask.float()
            masked_target = target_output * mask.float()
            
            all_predictions.extend(masked_pred[mask].cpu().numpy())
            all_targets.extend(masked_target[mask].cpu().numpy())
            
            loss = criterion(masked_pred, masked_target)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    r2 = r2_score(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    
    return avg_loss, r2, rmse, all_predictions, all_targets

def plot_test_cases(model, test_dataset, device, num_cases=5):
    indices = random.sample(range(len(test_dataset)), num_cases)
    
    for idx in indices:
        branch_input, trunk_input_polar, target, length = test_dataset[idx]
        
        print(f"\nCase {idx} data shapes:")
        print(f"trunk_input_polar shape: {trunk_input_polar.shape}")
        print(f"length: {length}")
        
        branch_input = branch_input.unsqueeze(0).to(device)
        trunk_input_polar = trunk_input_polar.unsqueeze(0).to(device)
        length = torch.tensor([length]).to(device)
        
        model.eval()
        with torch.no_grad():
            prediction = model(branch_input, trunk_input_polar, length)
            prediction = prediction[0, :length].cpu().numpy()
        
        valid_trunk_input = trunk_input_polar[0, :length].cpu().numpy()
        
        print(f"valid_trunk_input shape: {valid_trunk_input.shape}")
        print(f"prediction shape: {prediction.shape}")
        
        rho = valid_trunk_input[:, 0]
        theta = valid_trunk_input[:, 1]
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        
        valid_target = target[:length].numpy()
        
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        print(f"prediction shape: {prediction.shape}")
        print(f"valid_target shape: {valid_target.shape}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        vmin = np.min(prediction)
        vmax = np.max(prediction)
        scatter1 = ax1.scatter(x, y, c=prediction, cmap='viridis', vmin=vmin, vmax=vmax)
        ax1.set_title(f'Predicted Stress (Case {idx})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.axis('equal')
        plt.colorbar(scatter1, ax=ax1, label='Stress')
        
        error = prediction - valid_target
        error_max = np.max(np.abs(error))
        scatter2 = ax2.scatter(x, y, c=error, cmap='RdBu', vmin=-error_max, vmax=error_max)
        ax2.set_title('Prediction Error')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.axis('equal')
        plt.colorbar(scatter2, ax=ax2, label='Error')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nCase {idx} Statistics:")
        print(f"Number of points: {length.item()}")
        print(f"RMSE: {np.sqrt(np.mean((prediction - valid_target)**2)):.4f}")
        print(f"Max Absolute Error: {np.max(np.abs(error)):.4f}")
        print(f"Prediction range: [{np.min(prediction):.4f}, {np.max(prediction):.4f}]")
        print(f"Target range: [{np.min(valid_target):.4f}, {np.max(valid_target):.4f}]")

def get_ellipse_center(r, phi):
    """
    Convert polar coordinates (r, phi) to Cartesian coordinates (x, y)
    to get the center of the ellipse
    
    Args:
        r: distance from origin to ellipse center
        phi: angle in radians
    Returns:
        tuple: (x, y) coordinates of ellipse center
    """
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y

def convert_to_second_coordinate_system(points, r, phi):
    """
    Convert points from first coordinate system to second coordinate system
    centered at the ellipse
    
    Args:
        points: numpy array of shape (n_points, 2) containing x, y coordinates
        r: distance from origin to ellipse center
        phi: angle in radians
    Returns:
        numpy array: (n_points, 2) containing rho_2, angle_2 coordinates
    """
    center_x, center_y = get_ellipse_center(r, phi)
    
    translated_x = points[:, 0] - center_x
    translated_y = points[:, 1] - center_y
    
    rho_2 = np.sqrt(translated_x**2 + translated_y**2)
    angle_2 = np.arctan2(translated_y, translated_x)
    
    return np.column_stack((rho_2, angle_2))

def create_second_coordinate_inputs(trunk_inputs, branch_inputs):
    """
    Convert all trunk inputs to second coordinate system
    
    Args:
        trunk_inputs: list of (n_points, 2) arrays containing x, y coordinates
        branch_inputs: (n_samples, n_features) array containing parameters
    Returns:
        list: converted coordinates in second coordinate system
    """
    second_coords = []
    for i in range(len(trunk_inputs)):
        r = branch_inputs[i, 2]
        phi = branch_inputs[i, 3]
        
        coords = convert_to_second_coordinate_system(trunk_inputs[i], r, phi)
        second_coords.append(coords)
    
    return second_coords

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    folder_path = "train_data"
    branch_inputs, trunk_inputs, outputs, n_samples = load_data(folder_path)
    trunk_inputs1 = polar_coordinates(trunk_inputs)
    trunk_inputs2 = create_second_coordinate_inputs(trunk_inputs, branch_inputs)  # New coordinate system

    
    dataset = StressDataset(branch_inputs, trunk_inputs2, outputs)
    
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, collate_fn=collate_fn)
    
    model = DeepONet(branch_input_dim=7, trunk_input_dim=2, hidden_dim=64, output_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=500)
    
    plot_training_history(train_losses, val_losses)
    
    test_loss, r2_score, rmse, predictions, targets = evaluate_model(model, test_loader, criterion, device)
    print(f"\nTest Results:")
    print(f"Average Loss: {test_loss:.4f}")
    print(f"R虏 Score: {r2_score:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    plot_test_cases(model, test_dataset, device, num_cases=5)


