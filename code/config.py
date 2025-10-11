import argparse

def get_config():
    parser = argparse.ArgumentParser(description="IR Drop Prediction Configuration")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=600, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./processed_data/training_set/', help='Path to the dataset')
    parser.add_argument('--test_dir', type=str, default='./processed_data/testing_set/', help='Path to the dataset')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of splits for KFold cross validation')
    
    return parser.parse_args()
