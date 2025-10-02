
import pandas as pd
import torch

def standard_scaler(X):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    mean = torch.mean(X_tensor, dim=0)
    std = torch.std(X_tensor, dim=0)
    
    std = torch.where(std == 0, torch.ones_like(std), std)
    
    X_scaled = (X_tensor - mean) / std
    
    return X_scaled.numpy(), mean.numpy(), std.numpy()

def preprocess_titanic_data(file_path, device: str = 'cpu'):
    df = pd.read_csv(file_path)
    
    df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    
    df['Embarked'] = df['Embarked'].fillna('S')
    
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    df['IsAlone'] = ((df['SibSp'] + df['Parch']) == 0).astype(int)
    df['Age*Class'] = df['Age'] * df['Pclass']
    df['HighFare'] = (df['Fare'] > df['Fare'].median()).astype(int)
    df['AgeOfMan'] = (df['Age'] * df['Sex'])
    df['AgeOfWoman'] = (df['Age'] * (1 - df['Sex']))
    df['RichClass'] = ((df['Pclass'] == 1) & (df['Fare'] > df['Fare'].median())).astype(int)

    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    X = X.astype(float)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, device=device
    )
    
    X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)
    
    feature_names = X.columns.tolist()
    
    return X_train, X_test, y_train, y_test, feature_names


def preprocess_houses_data(file_path, device: str = 'cpu'):
    df = pd.read_csv(file_path)
    
    df['RoomsPerHousehold'] = df['Tot_Rooms'] / df['Households']
    df['BedroomsPerRoom'] = df['Tot_Bedrooms'] / df['Tot_Rooms']
    df['Spaciousness'] = df['Tot_Rooms'] / df['Tot_Bedrooms']
    df['MinDistanceToCity'] = df[['Distance_to_LA', 'Distance_to_SanDiego', 'Distance_to_SanJose', 'Distance_to_SanFrancisco']].min(axis=1)
    df['AgeToIncomeRatio'] = df['Median_Age'] / df['Median_Income']
    df['AvgDistanceToCities'] = df[['Distance_to_LA', 'Distance_to_SanDiego', 'Distance_to_SanJose', 'Distance_to_SanFrancisco']].mean(axis=1)
    df['RoomEfficiency'] = df['Tot_Bedrooms'] / df['Tot_Rooms']
    df['LuxuryIndex'] = df['Tot_Rooms'] / df['Population']
    df['Latitude_Longitude_Interaction'] = df['Latitude'] * df['Longitude']
    df['Is_Coastal'] = (df['Distance_to_coast'] < 10000).astype(int)
    df['Coastal_Income_Interaction'] = df['Is_Coastal'] * df['Median_Income']
    
    
    X = df.drop('Median_House_Value', axis=1)
    y = df['Median_House_Value']
    
    X_scaled, mean, std = standard_scaler(X.values)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, device=device
    )
    
    X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)
    
    feature_names = X.columns.tolist()
    
    return X_train, X_test, y_train, y_test, feature_names


def train_test_split(X, y, test_size=0.2, random_state=42, device='cpu'):
    torch.manual_seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)

    indices = torch.randperm(n_samples, device=device)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    train_indices_cpu = train_indices.cpu().numpy()
    test_indices_cpu = test_indices.cpu().numpy()
    
    return X.iloc[train_indices_cpu], X.iloc[test_indices_cpu], y.iloc[train_indices_cpu], y.iloc[test_indices_cpu]