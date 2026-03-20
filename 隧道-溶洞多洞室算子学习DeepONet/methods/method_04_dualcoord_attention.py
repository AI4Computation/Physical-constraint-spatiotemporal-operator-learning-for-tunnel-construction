import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import random
import os
import pandas as pd

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
            nn.Linear(hidden_dim, output_dim)
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
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DualStressDataset(Dataset):
    def __init__(self, branch_inputs, trunk_inputs1, trunk_inputs2, outputs):
        self.branch_inputs = torch.tensor(branch_inputs, dtype=torch.float32)
        self.trunk_inputs1 = [torch.tensor(x, dtype=torch.float32) for x in trunk_inputs1]
        self.trunk_inputs2 = [torch.tensor(x, dtype=torch.float32) for x in trunk_inputs2]
        self.outputs = [torch.tensor(x, dtype=torch.float32) for x in outputs]
        self.lengths = [len(x) for x in outputs]
        
    def __len__(self):
        return len(self.branch_inputs)
        
    def __getitem__(self, idx):
        return (
            self.branch_inputs[idx],
            self.trunk_inputs1[idx],
            self.trunk_inputs2[idx],
            self.outputs[idx],
            self.lengths[idx]
        )

def dual_collate_fn(batch):
    branch_inputs, trunk_inputs1, trunk_inputs2, outputs, lengths = zip(*batch)
    branch_inputs = torch.stack(branch_inputs)
    trunk_inputs1 = pad_sequence(trunk_inputs1, batch_first=True)
    trunk_inputs2 = pad_sequence(trunk_inputs2, batch_first=True)
    outputs = pad_sequence(outputs, batch_first=True)
    lengths = torch.tensor(lengths)
    return branch_inputs, trunk_inputs1, trunk_inputs2, outputs, lengths

def get_ellipse_center(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y

def convert_to_second_coordinate_system(points, r, phi):
    center_x, center_y = get_ellipse_center(r, phi)
    translated_x = points[:, 0] - center_x
    translated_y = points[:, 1] - center_y
    rho_2 = np.sqrt(translated_x**2 + translated_y**2)
    angle_2 = np.arctan2(translated_y, translated_x)
    return np.column_stack((rho_2, angle_2))

def create_second_coordinate_inputs(trunk_inputs, branch_inputs):
    second_coords = []
    for i in range(len(trunk_inputs)):
        r = branch_inputs[i, 2]
        phi = branch_inputs[i, 3]
        coords = convert_to_second_coordinate_system(trunk_inputs[i], r, phi)
        second_coords.append(coords)
    return second_coords

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
        angle_col = np.arctan2(y_col, x_col)  # 璁＄畻鏋佽,鍗曚綅涓哄姬搴︼紝angle = np.arctan2(ylist, xlist)
        
        polar_coords.append(np.column_stack((rho_col, angle_col)))
    return polar_coords

class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim, output_dim):
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet(branch_input_dim, hidden_dim, output_dim)
        self.trunk_net = TrunkNet(trunk_input_dim, hidden_dim, output_dim)

    def forward(self, branch_input, trunk_input, lengths):
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

class DualDeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim, output_dim):
        super(DualDeepONet, self).__init__()
        self.deeponet1 = DeepONet(branch_input_dim, trunk_input_dim, hidden_dim, output_dim)
        self.deeponet2 = DeepONet(branch_input_dim, trunk_input_dim, hidden_dim, output_dim)
        
        attention_dim = 15  # New attention dimension
        self.attention_proj1 = nn.Linear(1, attention_dim)
        self.attention_proj2 = nn.Linear(1, attention_dim)
        self.attention = nn.MultiheadAttention(attention_dim, num_heads=3)
        
        self.final_layer = nn.Linear(2, 1)
        
    def forward(self, branch_input, trunk_input1, trunk_input2, lengths):
        pred1 = self.deeponet1(branch_input, trunk_input1, lengths)
        pred2 = self.deeponet2(branch_input, trunk_input2, lengths)
        
        batch_size, max_points = pred1.shape
        pred1_reshaped = pred1.unsqueeze(2)  # (batch_size, max_points, 1)
        pred2_reshaped = pred2.unsqueeze(2)  # (batch_size, max_points, 1)
        
        proj1 = self.attention_proj1(pred1_reshaped)  # (batch_size, max_points, attention_dim)
        proj2 = self.attention_proj2(pred2_reshaped)  # (batch_size, max_points, attention_dim)
        
        proj1 = proj1.transpose(0, 1)
        proj2 = proj2.transpose(0, 1)
        
        attn_output, _ = self.attention(proj1, proj2, proj2)
        
        attn_output = attn_output.transpose(0, 1)  # (batch_size, max_points, attention_dim)
        
        combined_preds = torch.cat([pred1_reshaped, pred2_reshaped], dim=2)  # (batch_size, max_points, 2)
        
        output = self.final_layer(combined_preds)  # (batch_size, max_points, 1)
        output = output.squeeze(2)  # (batch_size, max_points)
        
        mask = torch.arange(max_points).expand(batch_size, max_points).to(output.device)
        mask = mask < lengths.unsqueeze(1)
        output = output * mask.float()
        
        return output

def train_dual_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=100):
    train_losses = []
    val_losses = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

    def compute_validation_loss(loader):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for branch_input, trunk_input1, trunk_input2, target_output, lengths in loader:
                branch_input = branch_input.to(device)
                trunk_input1 = trunk_input1.to(device)
                trunk_input2 = trunk_input2.to(device)
                target_output = target_output.to(device)
                lengths = lengths.to(device)

                prediction = model(branch_input, trunk_input1, trunk_input2, lengths)
                mask = torch.arange(prediction.size(1)).expand(prediction.size(0), -1).to(device)
                mask = mask < lengths.unsqueeze(1)
                loss = criterion(prediction * mask.float(), target_output * mask.float())
                total_loss += loss.item()
        return total_loss / len(loader)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for branch_input, trunk_input1, trunk_input2, target_output, lengths in train_loader:
            branch_input = branch_input.to(device)
            trunk_input1 = trunk_input1.to(device)
            trunk_input2 = trunk_input2.to(device)
            target_output = target_output.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            prediction = model(branch_input, trunk_input1, trunk_input2, lengths)
            
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
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
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

def evaluate_dual_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for branch_input, trunk_input1, trunk_input2, target_output, lengths in test_loader:
            branch_input = branch_input.to(device)
            trunk_input1 = trunk_input1.to(device)
            trunk_input2 = trunk_input2.to(device)
            target_output = target_output.to(device)
            lengths = lengths.to(device)

            prediction = model(branch_input, trunk_input1, trunk_input2, lengths)
            
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


def plot_dual_test_cases(model, test_dataset, device, num_cases=5):
    indices = random.sample(range(len(test_dataset)), num_cases)
    
    for idx in indices:
        branch_input, trunk_input1, trunk_input2, target, length = test_dataset[idx]
        
        branch_input = branch_input.unsqueeze(0).to(device)
        trunk_input1 = trunk_input1.unsqueeze(0).to(device)
        trunk_input2 = trunk_input2.unsqueeze(0).to(device)
        length = torch.tensor([length]).to(device)
        
        model.eval()
        with torch.no_grad():
            prediction = model(branch_input, trunk_input1, trunk_input2, length)
            prediction = prediction[0, :length].cpu().numpy()
        
        valid_trunk_input = trunk_input1[0, :length].cpu().numpy()
        rho = valid_trunk_input[:, 0]
        angle = valid_trunk_input[:, 1]
        x = rho * np.cos(angle)
        y = rho * np.sin(angle)
        
        valid_target = target[:length].numpy()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        vmin = min(np.min(prediction), np.min(valid_target))
        vmax = max(np.max(prediction), np.max(valid_target))
        
        scatter1 = ax1.scatter(x, y, c=prediction, cmap='viridis', vmin=vmin, vmax=vmax)
        ax1.set_title(f'Predicted Stress (Case {idx})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.axis('equal')
        plt.colorbar(scatter1, ax=ax1, label='Stress')
        
        scatter2 = ax2.scatter(x, y, c=valid_target, cmap='viridis', vmin=vmin, vmax=vmax)
        ax2.set_title('Target Stress')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.axis('equal')
        plt.colorbar(scatter2, ax=ax2, label='Stress')
        
        error = prediction - valid_target
        error_max = np.max(np.abs(error))
        scatter3 = ax3.scatter(x, y, c=error, cmap='RdBu', vmin=-error_max, vmax=error_max)
        ax3.set_title('Prediction Error')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.axis('equal')
        plt.colorbar(scatter3, ax=ax3, label='Error')
        
        plt.tight_layout()
        plt.show()
        


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    folder_path = "train_data"  # Adjust this to your data path
    branch_inputs, trunk_inputs, outputs, n_samples = load_data(folder_path)
    
    trunk_inputs1 = polar_coordinates(trunk_inputs)  # Original coordinate system
    trunk_inputs2 = create_second_coordinate_inputs(trunk_inputs, branch_inputs)  # New coordinate system
    
    dataset = DualStressDataset(branch_inputs, trunk_inputs1, trunk_inputs2, outputs)
    
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=3,
        shuffle=True,
        collate_fn=dual_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=dual_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=dual_collate_fn
    )
    
    model = DualDeepONet(
        branch_input_dim=7,  # [k, ratio, r, phi, theta, a, b]
        trunk_input_dim=2,   # [rho/rho_2, angle/angle_2]
        hidden_dim=64,
        output_dim=32
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Starting training...")
    train_losses, val_losses = train_dual_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        epochs=100
    )
    
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
    
    print("\nEvaluating model on test set...")
    test_loss, r2, rmse, predictions, targets = evaluate_dual_model(
        model,
        test_loader,
        criterion,
        device
    )
    
    print(f"\nTest Results:")
    print(f"Average Loss: {test_loss:.4f}")
    print(f"R虏 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    print("\nPlotting test cases...")
    plot_dual_test_cases(model, test_dataset, device, num_cases=5)
    
    print("\nSaving model...")
    torch.save(model.state_dict(), 'dual_deeponet_model.pth')
    print("Model saved successfully!")



