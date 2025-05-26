import json
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def read_txt_files(folder_path) -> dict:
    class_counts = {}

    print(f"Total number of images: {len([name for name in os.listdir(folder_path) if name.endswith('.txt')])}")
    # 遍历文件夹中的所有txt文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                # 读取每行数据并统计类别数量
                for line in file:
                    if line.strip():
                        category = int(line.split()[0])
                        class_counts[category] = class_counts.get(category, 0) + 1

    counter = 0
    for i in range(len(class_counts)):
        counter += class_counts[i]
    
    print(f"Total number of annotations: {counter}")
    return class_counts

def plot_bar_chart(
    class_counts, 
    class_mapping: dict, 
    plot_title: str = "Número de anotaciones por clase", 
    save_path: str = "plot.eps"
):
    # Sort categories and counts by the key (integer label)
    sorted_categories = sorted(class_counts.keys())
    sorted_counts = [class_counts[category] for category in sorted_categories]

    # Map sorted integer labels to class names
    sorted_class_names = [class_mapping[category] for category in sorted_categories]

    # Generate a color map with as many colors as there are categories
    color_map = plt.get_cmap('hsv', len(sorted_categories)+1)
    colors = color_map(range(len(sorted_categories)))

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_class_names, sorted_counts, color=colors)
    plt.xlabel('Categoría')
    plt.ylabel('Cantidad de anotaciones')
    plt.title(plot_title)
    plt.xticks(rotation=45)
    plt.ylim(0, max(sorted_counts) + 200 if max(sorted_counts) > 2000 else 2000)
    plt.tight_layout()

    plt.savefig(save_path, format='eps', dpi=300)
    plt.close()  # Close the plot to free up memory

def main(split: str = "full", dataset_path: str = None) -> dict:
    
    # Get the current directory where the Python script is located
    # If path is not provided, use the current directory
    if dataset_path is None:
        dataset_path = os.path.dirname(os.path.abspath(__file__)) 
        dataset_path = os.path.join(path, "datasets/trash_detection")
    print(f"Dataset path: {dataset_path}")

    # Append a relative path to the current directory
    if split == "full":
        labels_path = os.path.join(dataset_path, "labels")
    else:
        labels_path = os.path.join(dataset_path, f'labels/{split}')  # Replace with your relative path
    
    print(f"Labels path: {labels_path}")

    # Check if the path exists
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"The specified path does not exist: {labels_path}")

    class_counts = read_txt_files(labels_path)

    # Get class names from json file
    classes_json_path = os.path.join(dataset_path, "classes.json")
    if not os.path.exists(classes_json_path):
        raise FileNotFoundError(f"The specified path does not exist: {classes_json_path}")
    with open(classes_json_path, 'r') as f:
        class_mapping = json.load(f)
        # Convert class_mapping to a dictionary with integer keys
        class_mapping = {int(k): v for k, v in class_mapping.items()}
    
    save_dir = os.path.join(dataset_path, "my_dataset_class_count_" + split + ".eps")
    plot_bar_chart(
        class_counts, 
        class_mapping=class_mapping, 
        plot_title='Número de anotaciones por clase en el conjunto completo' if split == 'full' else f"Número de anotaciones por clase en la partición '{split}'",
        save_path=save_dir)    
    
    return class_counts

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Count dataset classes and generate bar charts.")
    parser.add_argument("-d", "--dataset_path", type=str, required=True, help="Path to the dataset folder containing label files.")
    parser.add_argument("-s", "--splits", type=str, default="full", help="Comma-separated list of split to analyze (full, train, val, test).")
    args = parser.parse_args()
    
    abs_dataset_path = os.path.abspath(os.path.normpath(args.dataset_path))
    # print(abs_dataset_path)

    class_counts = []
    splits = args.splits.split(",")
    # Check if the provided splits are valid
    valid_splits = ["full", "train", "val", "test"]
    for split in splits:
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Valid splits are: {valid_splits}")

    for split in splits:
        class_counts.append(main(split, dataset_path=abs_dataset_path))
    
    # Plot proportions
    for i in range(len(class_counts)):
        if i == 0: continue
        for j in class_counts[i].keys():
            class_counts[i][j] = f"{float(class_counts[i][j] / class_counts[0][j])*100.0:.2f}"

    for aux in class_counts:
        print(aux)
    