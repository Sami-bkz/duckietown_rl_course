import pathlib
import pandas as pd
import numpy as np

# Get the current directory
current_dir = pathlib.Path(__file__).parent.absolute()

# Find all parquet files in the directory
parquet_files = list(current_dir.glob("*.parquet"))

# Check if there are any parquet files
if not parquet_files:
    print("No .parquet files found in the current directory.")
    exit()

# Display available files
print("\nChoose the file to load among the following ones:")
for idx, file in enumerate(parquet_files, 1):
    print(f"{idx} - {file.name}")

# Get user input
while True:
    try:
        choice = int(input("\nEnter the number of the file you want to load: ")) - 1
        if 0 <= choice < len(parquet_files):
            selected_file = parquet_files[choice]
            break
        else:
            print("Invalid selection. Please choose a valid number.")
    except ValueError:
        print("Please enter a valid integer.")

# Load the selected parquet file
df = pd.read_parquet(selected_file, engine="pyarrow")

obs_shape = (330, 640, 3)

# Convert flattened lists back into NumPy arrays with the correct shape
df["s"] = df["s"].apply(lambda x: np.array(x, dtype=np.uint8).reshape(obs_shape))
df["next_s"] = df["next_s"].apply(lambda x: np.array(x, dtype=np.uint8).reshape(obs_shape))

# Example: Accessing the first observation and its corresponding next state
first_obs = df.iloc[0]["s"]
first_action = df.iloc[0]["a"]
first_reward = df.iloc[0]["r"]
first_done = df.iloc[0]["d"]
first_next_obs = df.iloc[0]["next_s"]
actions = [df.iloc[i]["a"] for i in range(20)]

# Print the shapes to verify
print("First observation shape:", first_obs.shape)
print("First next observation shape:", first_next_obs.shape)
print("First action:", first_action)
print("First reward:", first_reward)
print("First done flag:", first_done)
print("actions:", actions)

# Display success message
print(f"\nLoaded file: {selected_file.name}")
