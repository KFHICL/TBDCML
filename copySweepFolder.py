# %%

import os
import shutil
from datetime import datetime

def copy_and_rename_folder(src_folder, src_python_folder, dest_folder_names, num_jobs_list, datasetNames, crossValidation, jobDetails, jobParameters):
    for name, num_jobs, dataset, use_cross_validation, jobDetails, jobParameters in zip(dest_folder_names, num_jobs_list, datasetNames, crossValidation, jobDetails, jobParameters):
        # Create the destination folder in the same folder as the src folder
        dest_folder = os.path.join(os.path.dirname(src_folder), name)
        print(dest_folder)
        shutil.copytree(src_folder, dest_folder)
        
        # Copy Python files from the source Python folder to the destination folder
        for file in os.listdir(src_python_folder):
            if file in ["CrossValidation.py", "BenchMarks.py"]:
                shutil.copy(os.path.join(src_python_folder, file), dest_folder)
        
        # Rename files within the copied folder
        for root, dirs, files in os.walk(dest_folder):
            for file in files:
                
                # Modify .sh files
                if file.endswith(".sh"):
                    sh_file_path = os.path.join(root, file)
                    with open(sh_file_path, "r") as f:
                        lines = f.readlines()
                    
                    with open(sh_file_path, "w",newline="\n") as f:
                        for line in lines:
                            if "jobName=TEMPLATE" in line:
                                line = line.replace("jobName=TEMPLATE", f"jobName={name}")
                            if "#PBS -J 1-TEMPLATE_NUM_JOBS" in line:
                                line = line.replace("#PBS -J 1-TEMPLATE_NUM_JOBS", f"#PBS -J 1-{num_jobs}")
                            if "python3 $HOME/IndividualProject/CNNTraining/$jobName/CrossValidation.py" in line:
                                if not use_cross_validation:
                                    line = f"# {line}"
                            if "python3 $HOME/IndividualProject/CNNTraining/$jobName/BenchMarks.py" in line:
                                if use_cross_validation:
                                    line = f"# {line}"
                            f.write(line)

                
                
                # Modify CSV files
                if file.endswith(".csv"):
                    csv_file_path = os.path.join(root, file)
                    with open(csv_file_path, "r") as f:
                        lines = f.readlines()
                    
                    # Ensure there are at least two rows to copy
                    if len(lines) > 1:
                        header = lines[0]
                        row_to_copy = lines[1]
                        new_rows = []
                        current_index = 1
                        
                        # Find the index of the "Dataset" column
                        header_columns = header.strip().split(",")
                        dataset_index = header_columns.index("Dataset")
                        
                        # Update the dataset in the row being copied
                        row_to_copy = row_to_copy.split(",")
                        row_to_copy[dataset_index] = dataset
                        row_to_copy = ",".join(row_to_copy)
                        
                        
                        for i in range(num_jobs - 1):
                            current_index += 1
                            new_row = row_to_copy.split(",")
                            new_row[0] = str(current_index)  # Update the "Index" column
                            new_rows.append(",".join(new_row))
                        allrows = [row_to_copy] + new_rows

                        # Modify columns corresponding to jobDetails
                        for detail in jobDetails:
                            if isinstance(detail, str):  # Handle case where jobDetails is a single string
                                detail = [detail]
                            for param_name in detail:
                                if param_name in header_columns:
                                    param_index = header_columns.index(param_name)
                                    for i, row in enumerate(allrows):
                                        row_split = row.split(",")
                                        if "modelDepth" in name:
                                            # Special handling for "modelDepth"
                                            param_value = jobParameters[i % len(jobParameters)]
                                            depth = int(param_name[-1])
                                            layer_index = header_columns.index(param_name)
                                            if depth <= param_value:
                                                row_split[layer_index] = "1"
                                            else:
                                                row_split[layer_index] = "0"
                                        else:
                                            # Assign values from jobParameters cyclically if there are more rows than parameters
                                            row_split[param_index] = str(jobParameters[i % len(jobParameters)])
                                        allrows[i] = ",".join(row_split)
                        
                        # Write the updated CSV file
                        with open(csv_file_path, "w") as f:
                            f.write(header)  # Write the header
                            f.writelines(allrows)  # Write all rows
                            # f.write(row_to_copy + "\n")  # Add the copied row
                            # f.writelines("\n".join(new_rows))  # Add new rows
                if "TEMPLATE" in file:
                    old_file_path = os.path.join(root, file)
                    new_file_name = file.replace("TEMPLATE", name)
                    new_file_path = os.path.join(root, new_file_name)
                    os.rename(old_file_path, new_file_path)
# Example usage
# template_folder = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\TEMPLATE"  # Replace with the path to the folder you want to copy
template_folder = r"\\rds.imperial.ac.uk\rds\user\kfh23\home\IndividualProject\CNNTraining\TEMPLATE_run2"  # Another run of everything but this time with 4-layer deep models
runName = '_run2'

code_dir = os.path.dirname(os.path.abspath(__file__))
while os.path.basename(code_dir) != "Individual Project":
    code_dir = os.path.dirname(code_dir)
source_python_folder = os.path.join(code_dir, "Code", "TBDCML_Clone", "TBDCML")

# Define the job folders to be created
jobNames = ["trainValSplit", 
            "batchSize", 
            "kernelSize", 
            "optimizer", 
            "activationFunction", 
            "loss", 
            "dropout", 
            "initialLr", 
            "lrDecay", 
            'maxPool', 
            "batchNorm", 
            # "modelDepth", 
            "dataAug", 
            "skinConnections", 
            "decoderAct", 
            "epsilon"]  # sweeping job names

jobNames = [s + runName for s in jobNames]
num_jobs_list = [3,             # trainValSplit
                 7,              # batchSize
                 7,          # kernelSize
                 3,              # optimizer
                 7,              # activationFunction
                 3,          # loss
                 7,          # dropout
                 5,          # initialLr
                 4,      # lrDecay
                 2,          # maxPool
                 2,              # batchNorm
                #  6,          # modelDepth
                 2,          # dataAug
                 2,              # skinConnections
                 2,              # decoderAct
                 7,      # epsilon
                 ]  
# Replace with the number of jobs for each project

# Define a new list with one or more strings for each jobName
jobDetails = [
    ["valSize"],  # trainValSplit
    ["batchSize"],  # batchSize
    ["layer1Kernel", "layer2Kernel", "layer3Kernel", "layer4Kernel", "layer5Kernel", "layer6Kernel"],  # kernelSize
    ["optimizer"],  # optimizer
    ["conv1Activation", "conv2Activation", "conv3Activation", "conv4Activation", "conv5Activation", "conv6Activation"],  # activationFunction
    ["loss"],  # loss
    ["dropout"],  # dropout
    ["initial_lr"],  # initialLr
    ["lr_decay_rate"],  # lrDecay
    ["pooling"],  # maxPool
    ["batchNorm"],  # batchNorm
    # ["layer2", "layer3", "layer4", "layer5", "layer6"],  # modelDepth
    ["dsAugmentation"],  # dataAug
    ["skipConnections"],  # skinConnections
    ["ActivationUp"],  # decoderAct
    ["epsilon"],  # epsilon
]
# Set parameters for each job (the param is the one in jobDetails)
jobParameters = [
    [0.1, 0.2, 0.3],  # trainValSplit
    [1, 2, 4, 8, 16, 32, 64],  # batchSize
    [1, 2, 3, 4, 5, 6, 7],  # kernelSize
    ["Adam", "Nadam", "Adadelta"],  # optimizer
    ["relu", "tanh", "softplus", "elu", "leaky_relu", "silu", "gelu"],  # activationFunction
    ["MSE", "MAE", "Custom"],  # loss
    [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5],  # dropout
    [0.0001, 0.0005, 0.001, 0.005, 0.01],  # initialLr
    [1, 0.5, 0.1, 0.01],  # lrDecay
    [1, 0],  # maxPool
    [1, 0],  # batchNorm
    # [1, 2, 3, 4, 5, 6],  # modelDepth
    [1, 0],  # dataAug
    [1, 0],  # skinConnections
    [1, 0],  # decoderAct
    [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],  # epsilon
]

# Formate automatically
datasetNames = ["LFC18", "MC24", "MC24x"]  # Replace with the dataset names for each project

jobNames = jobNames*len(datasetNames)  # Repeat the job names for each dataset
jobDetails = jobDetails*len(datasetNames) # Repeat the job details for each job
jobParameters = jobParameters*len(datasetNames)  # Repeat the job parameters for each job
num_jobs_list = num_jobs_list*len(datasetNames)  # Repeat the number of jobs for each dataset

datasetNames = [i for i in datasetNames for _ in range(int(len(jobNames)/len(datasetNames)) )] # Repeat the dataset names for each job


# Reformat the job names to include the date and dataset
today_date = datetime.now().strftime("%Y%m%d")
jobName_list = [f"{today_date}_{dataset}_{job}" for dataset, job in zip(datasetNames, jobNames)]

crossValidation = [False for i in jobName_list]  # Define which script to call for each project


copy_and_rename_folder(template_folder, source_python_folder, jobName_list, num_jobs_list, datasetNames, crossValidation, jobDetails, jobParameters)