import joblib
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from first_attempt_tech_task import pd_dataset_utils


class HouseDataset(Dataset):
    def __init__(self, features, target):
        super().__init__()

        if len(features) != len(target):
            raise ValueError("Features and Target size doesn't match.")
        self.len = len(features)
        self.features = features
        self.target = target

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        X = torch.FloatTensor(self.features.iloc[index].values)
        y = torch.FloatTensor([self.target.iloc[index]])
        return X, y


class HouseClassifier(nn.Module):
    def __init__(self, input_size):
        super(HouseClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)


def filter_data(df):
    df = pd_dataset_utils.encode_pd_df(df)
    df = pd_dataset_utils.handle_nulls(df)
    df = pd_dataset_utils.drop_duplicates(df)
    # handle outliers
    return df


6


def train(dataset_path, num_epochs=3):
    label_encoder = LabelEncoder()
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    df = pd.read_csv(dataset_path)
    df = pd_dataset_utils.handle_nulls(df)
    df = pd_dataset_utils.drop_duplicates(df)
    df = pd_dataset_utils.encode_pd_df(df, label_encoder)
    # TODO handle outliers

    target = df.price
    features = df.drop("price", axis=1)

    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1)).flatten()

    features = pd.DataFrame(features_scaled, columns=features.columns)
    target = pd.Series(target_scaled)

    X_train, X_val, y_train, y_val = train_test_split(
        target, features, test_size=0.2, random_state=42
    )

    train_dataset = HouseDataset(features=y_train, target=X_train)
    val_dataset = HouseDataset(features=y_val, target=X_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = HouseClassifier(input_size=len(features.columns))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    criterion = nn.BCEWithLogitsLoss()  # nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        # traning
        model.train()
        train_loss = 0.0
        for batch_features, batch_target in train_loader:
            batch_features = batch_features.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            output = model(batch_features)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validataion
        model.eval()
        val_los = 0
        with torch.no_grad():
            for batch_features, batch_target in val_loader:
                batch_features = batch_features.to(device)
                batch_target = batch_target.to(device)

                outputs = model(batch_features)
                loss = criterion(outputs, batch_target)
                val_los += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_los / len(val_loader)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "saved_model/best_model.pth")
            joblib.dump(label_encoder, "saved_model/label_encoder.pkl")
            joblib.dump(feature_scaler, "saved_model/feature_scaler.pkl")
            joblib.dump(target_scaler, "saved_model/target_scaler.pkl")


def inference(model_path, label_encoder, feature_scaler, data_path):
    model = HouseClassifier(input_size=545)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    label_encoder = joblib.load(label_encoder)
    feature_scaler = joblib.load(feature_scaler)

    df = pd.read_csv(data_path)
    df = pd_dataset_utils.encode_pd_df(df, label_encoder=label_encoder)


if __name__ == "__main__":
    data_path = r"first_attempt_tech_task\dataset\Housing.csv"
    train(data_path, num_epochs=5)
