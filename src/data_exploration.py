import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def explore_data():
    # Load the data
    train_df = pd.read_csv('../dataset/train.csv')
    test_df = pd.read_csv('../dataset/test.csv')

    # Display basic information
    print("Train dataset info:")
    print(train_df.info())
    print("\nTest dataset info:")
    print(test_df.info())

    # Analyze distribution of entity types
    plt.figure(figsize=(12, 6))
    train_df['entity_name'].value_counts().plot(kind='bar')
    plt.title('Distribution of Entity Types')
    plt.tight_layout()
    plt.savefig('../output/entity_types_distribution.png')
    plt.close()

    # Analyze distribution of units (if available in train_df)
    if 'entity_value' in train_df.columns:
        units = train_df['entity_value'].str.split().str[-1]
        plt.figure(figsize=(12, 6))
        units.value_counts().plot(kind='bar')
        plt.title('Distribution of Units')
        plt.tight_layout()
        plt.savefig('../output/units_distribution.png')
        plt.close()

    # Additional analysis: Group distribution
    plt.figure(figsize=(12, 6))
    train_df['group_id'].value_counts().plot(kind='bar')
    plt.title('Distribution of Group IDs')
    plt.tight_layout()
    plt.savefig('../output/group_id_distribution.png')
    plt.close()

    # Print some sample data
    print("\nSample data from train dataset:")
    print(train_df.head())

    print("\nSample data from test dataset:")
    print(test_df.head())

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('../output', exist_ok=True)
    explore_data()