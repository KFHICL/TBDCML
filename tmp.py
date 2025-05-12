# %%
import os

def find_folders_with_date(path, date):
    matching_folders = []
    count = 0
    # for root, dirs, files in os.walk(path):
    for dirs in os.listdir(path):
        print(dirs)
        if date in dirs:
            matching_folders.append(os.path.join(path, dirs))
                
    return matching_folders

# Example usage
path_to_search = r'\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining'
date_to_find = "20250512"
folders = find_folders_with_date(path_to_search, date_to_find)
# %%
# Generate output for spreadsheet
output = []
prefix_to_remove = "\\\\rds.imperial.ac.uk\\rds\\user\\kfh23\\home\\"
for folder in folders:
    folder_short = folder.replace(prefix_to_remove, "").replace("\\", "/")
    commands = ["cd", f"cd {folder_short}"]
    for file in os.listdir(folder):
        if file.endswith(".sh"):
            commands.append(f"qsub {file} \\")
    commands = ["cd", f"cd {folder_short}"]
    for file in os.listdir(folder):
        if file.endswith(".sh"):
            commands.append(f"qsub {file}")
    output.append("\n".join(commands))

# Print output
print("\n\n".join(output))
