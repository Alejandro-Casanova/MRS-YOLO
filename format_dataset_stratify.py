import json
import shutil
from typing import Tuple
import pandas as pd
import os
import glob
import argparse

# https://www.reddit.com/r/computervision/comments/im0rji/how_to_split_custom_dataset_for_training_and/

def generate_csv(dataset_dir: str = "datasets/trash_detection"):
    # Specify the directory containing images and labels
    image_dir = os.path.join(dataset_dir, 'images')
    label_dir = os.path.join(dataset_dir, 'labels')

    # Check if the directories exist    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory {image_dir} does not exist.")
    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"Label directory {label_dir} does not exist.")

    # Get all image file paths
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) \
        + glob.glob(os.path.join(image_dir, "*.png")) \
        + glob.glob(os.path.join(image_dir, "*.jpeg"))

    # Get all label file paths
    label_paths = glob.glob(os.path.join(label_dir, "*.txt"))
    print(f"Found {len(label_paths)} label files.")
    if len(label_paths) == 0:
        raise FileNotFoundError(f"No label files found in {label_dir}.")

    # Create a dictionary to store data
    data = {
        "file_path": [],
        "label_path": [],
        "labels": [],
    }

    # Initialize a set to store class names
    classes_set = set()

    print("Processing label files to produce CSV...")

    # Process each label file
    for label_path in label_paths:
        with open(label_path, "r") as file:
            lines = file.readlines()
            classes_in_image = {}
            for line in lines:
                if line.strip():
                    class_id = line.split()[0]
                    # Increment class count
                    if class_id in classes_in_image:
                        classes_in_image[class_id] += 1
                    else:
                        classes_in_image[class_id] = 1
                    classes_set.add(class_id)
            
            # Save file path and labels to data
            data["label_path"].append(label_path)
            corresponding_image_path = label_path.replace(label_dir, image_dir).replace('.txt', '.jpg')
            data["file_path"].append(corresponding_image_path)
            data["labels"].append(classes_in_image)

    # Add class columns to data
    for class_name in sorted(classes_set):
        data[class_name] = []

    # Fill in class columns
    for labels in data["labels"]:
        for class_name in sorted(classes_set):
            if class_name in labels:
                data[class_name].append(labels[class_name])
            else:
                data[class_name].append(0)

    # Remove labels column (no longer needed)
    data.pop("labels")

    # Create DataFrame from the data dictionary
    df = pd.DataFrame(data)

    print(f"Finished processing dataset. Found {len(df)} images and {len(classes_set)} classes.")
    print(f"Classes: {sorted(classes_set)}")
    return df

    # Save the DataFrame to a CSV file
    df.to_csv("yolo_image_data.csv", index=False)

def check_stratified_condition(df: pd.DataFrame, desire_set: dict, class_names: list, condition: int, tolerance: int, ratio: float, random_state: int):
    tmp_df = df.sample(frac=ratio, replace=False, random_state=random_state)
    for name in class_names:
        if tmp_df[name].sum() < desire_set[name] + tolerance \
        and tmp_df[name].sum() > desire_set[name] - tolerance:
            condition += 1
    return tmp_df, condition

def stratify_sample(
    # desire_set: Tuple[dict] = None, 
    # image_id_classname_path: str, 
    ratios: Tuple[float] = None,
    input_dir: str = "datasets/trash_detection",
    output_dir: str = "datasets/trash_detection_split",
    random_seed: int = 42,
    class_mapping: dict = None  
):
    
    train_ratio, val_ratio, test_ratio = ratios
    # desire_train_set, desire_val_set, desire_test_set = desire_set  # goal of stratified splitting
    # df = pd.read_csv(image_id_classname_path)
    df = generate_csv(input_dir)
    # print(df.head())
    df_classes = df.drop(columns=["file_path", "label_path"])
    # print(df_classes.head())
    df_classes = df_classes.sum(axis=0)
    # print(df_classes)

    desire_train_set = dict()
    desire_val_set = dict()
    desire_test_set = dict()
    for class_name in df_classes.keys():
        desire_train_set[class_name] = int(train_ratio * df_classes[class_name])
        desire_val_set[class_name] = int(val_ratio * df_classes[class_name])
        desire_test_set[class_name] = int(test_ratio * df_classes[class_name])

    # print(desire_train_set)
    # print(desire_val_set)
    # print(desire_test_set)
    
    tolerance = 19
    class_names = list(desire_train_set.keys())
    condition = 0
    iter_limit = 10000
    iter_count = 0
    print('\nStarting iterating stratifying sampling...')

    tmp_train_df = None
    tmp_val_df = None
    tmp_test_df = None

    max_progress = 0

    while condition < len(class_names) * 3:
        iter_count += 1
        random_state = random_seed + iter_count
        _df = df.copy()

        # print(f'Iteration: {iter_count -1} / {iter_limit}, condition: {condition}, tolerance: {tolerance}')

        # Plot progress bar
        num_hashes = 20
        current_progress = (condition / (len(class_names) * 3)) * 100
        max_progress = current_progress if current_progress > max_progress else max_progress
        iteration_progress = (iter_count / iter_limit) * num_hashes    
        print(f"\rProgress: [{int(iteration_progress) * '#'}{int(num_hashes - iteration_progress) * ' '}] {iteration_progress / num_hashes * 100:.2f}% | Closeness to success: {max_progress:.2f}% ", end='')

        if iter_count >= iter_limit:
            iter_count = 0
            tolerance += 1
            print('Exceeded iteration limit... updated tolerance to:', tolerance)

        condition = 0
        # train set
        tmp_train_df, condition = check_stratified_condition(_df, desire_train_set,
                                                             class_names, condition,
                                                             tolerance, train_ratio,
                                                             random_state=random_state)
        
        if condition < len(class_names):
            continue # if not satisfied, continue to next iteration

        # val set, ratio changed due to train set is taken out
        _tmp_val_df = _df[~_df.index.isin(list(tmp_train_df.index))]
        tmp_val_df, condition = check_stratified_condition(_tmp_val_df, desire_val_set,
                                                           class_names, condition,
                                                           tolerance, val_ratio / (val_ratio + test_ratio),
                                                           random_state=random_state)
        
        if condition < len(class_names) * 2:
            continue # if not satisfied, continue to next iteration

        # test set
        _tmp_test_df = _tmp_val_df[~_tmp_val_df.index.isin(list(tmp_val_df.index))]
        tmp_test_df, condition = check_stratified_condition(_tmp_test_df, desire_test_set,
                                                            class_names, condition,
                                                            tolerance, 1,
                                                            random_state=random_state)
    print(f'Condition satisfied tolerance: {tolerance}')

    # print(tmp_train_df.head())
    print("\n")
    print(f"Train images: {len(tmp_train_df.index)}")
    print(f"Val images: {len(tmp_val_df.index)}")
    print(f"Test images: {len(tmp_test_df.index)}")

    train_split_instances = tmp_train_df.drop(columns=["file_path", "label_path"]).sum(axis=0)
    val_split_instances = tmp_val_df.drop(columns=["file_path", "label_path"]).sum(axis=0)
    test_split_instances = tmp_test_df.drop(columns=["file_path", "label_path"]).sum(axis=0)

    # print("Instances in train split: ", train_split_instances)
    # print("Instances in val split: ", val_split_instances)
    # print("Instances in test split: ", test_split_instances)

    # Calculate split error
    train_split_error = abs(train_split_instances - pd.Series(desire_train_set))
    train_split_error /= pd.Series(desire_train_set)
    val_split_error = abs(val_split_instances - pd.Series(desire_val_set))
    val_split_error /= pd.Series(desire_val_set)
    test_split_error = abs(test_split_instances - pd.Series(desire_test_set))
    test_split_error /= pd.Series(desire_test_set)

    # Create a DataFrame to display split instance numbers and errors
    error_table = pd.DataFrame({
        "Metric": ["Train Instances", "Train Error (%)", "Val Instances", "Val Error (%)", "Test Instances", "Test Error (%)"]
    })

    for class_name in class_names:
        error_table[class_name] = [
            int(train_split_instances[class_name]),
            round(train_split_error[class_name] * 100, 2),
            int(val_split_instances[class_name]),
            round(val_split_error[class_name] * 100, 2),
            int(test_split_instances[class_name]),
            round(test_split_error[class_name] * 100, 2)
        ]

    # Add averaged values across splits
    error_table["Averaged Across Classes"] = error_table.iloc[:, 1:].mean(axis=1)

    # Calculate total average error
    total_average_error = error_table.loc[
        error_table["Metric"].isin(["Train Error (%)", "Val Error (%)", "Test Error (%)"]),
        "Averaged Across Classes"
    ].mean()

    # Print the error table
    print("\nError Table (in %):")
    print(error_table)
    print(f"\nTotal Average Error: {total_average_error:.2f}%\n")

    # Replace class names with their corresponding labels
    for i, class_name in enumerate(class_names):
        if class_name in class_mapping:
            error_table.rename(columns={class_name: class_mapping[class_name]}, inplace=True)
            
    # Save the error table as a LaTeX file
    latex_file_path = os.path.join(output_dir, "error_table.tex")
    print("\nSaving Error Table as LaTeX file...")    
    with open(latex_file_path, "w") as latex_file:
        latex_file.write(error_table.to_latex(index=False, float_format="%.2f"))
    print(f"Error Table saved to {latex_file_path}")
    
    # Convert to list of tuples
    train_files = list(zip(tmp_train_df['file_path'], tmp_train_df['label_path']))
    val_files = list(zip(tmp_val_df['file_path'], tmp_val_df['label_path']))
    test_files = list(zip(tmp_test_df['file_path'], tmp_test_df['label_path']))

    # Define YOLO folder structure
    image_dir_out = os.path.join(output_dir, 'images')
    labels_dir_out = os.path.join(output_dir, 'labels')

    # Create YOLO folder structure if not already present
    for dir_path in [image_dir_out, labels_dir_out]:
        os.makedirs(os.path.join(dir_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'val'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'test'), exist_ok=True)

    # Helper function to copy files to their respective directories
    def copy_files(file_list, subset_dir):
        for img_path, ann_path in file_list:
            shutil.copy(img_path, os.path.join(image_dir_out, subset_dir, os.path.basename(img_path)))
            shutil.copy(ann_path, os.path.join(labels_dir_out, subset_dir, os.path.basename(ann_path)))

    # Copy files to train, val, and test directories
    print("Copying images...")
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")
    print("Finished!")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Stratify and split dataset for YOLO training.")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Path to the input dataset directory.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path to the output directory for the split dataset.")
    parser.add_argument("-s", "--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Abort if the output directory already exists
    if os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' already exists. Aborting to prevent overwriting.")
        exit(1)
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Get class names from input directory (json file)
    class_mapping = {}
    with open(os.path.join(input_dir, 'classes.json'), 'r') as f:
        class_mapping = json.load(f)


    stratify_sample(
        ratios=(0.70, 0.15, 0.15),
        input_dir=input_dir,
        output_dir=output_dir,
        random_seed=args.random_seed,
        class_mapping=class_mapping
    )