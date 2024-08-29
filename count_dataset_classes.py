import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

def read_txt_files(folder_path):
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
    plot_title: str = "Class Counts", 
    save_path: str = "plot.png"
):
    # Sort categories and counts by the key (integer label)
    sorted_categories = sorted(class_counts.keys())
    sorted_counts = [class_counts[category] for category in sorted_categories]

    # Map sorted integer labels to class names
    sorted_class_names = [class_mapping[category] for category in sorted_categories]

    # Generate a color map with as many colors as there are categories
    colormap = plt.get_cmap('hsv', len(sorted_categories)+1)
    colors = colormap(range(len(sorted_categories)))

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_class_names, sorted_counts, color=colors)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title(plot_title)
    plt.xticks(rotation=45)
    plt.ylim(0, max(sorted_counts) + 200 if max(sorted_counts) > 2000 else 2000)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()  # Close the plot to free up memory

def main(split: str = "full"):
    SPLIT = split
    # Get the current directory where the Python script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Append a relative path to the current directory
    if SPLIT == "full":
        relative_path = f'datasets/trash_detection/labels'
    else:
        relative_path = f'datasets/trash_detection_split/labels/{SPLIT}'  # Replace with your relative path
    
    full_path = os.path.join(current_dir, relative_path)
    print(f"Full path: {full_path}")

    class_counts = read_txt_files(full_path)
    print(class_counts)

    # Class names
    class_mapping = {0: 'Not recyclable', 1: 'Food waste', 2: 'Glass', 3: 'Textile', 4: 'Metal', 5: 'Wooden', 6: 'Leather', 7: 'Plastic', 8: 'Ceramic', 9: 'Paper'}
    
    save_dir = os.path.join(current_dir, "my_dataset_class_count_" + SPLIT + ".png")
    plot_bar_chart(
        class_counts, 
        class_mapping=class_mapping, 
        plot_title='Number of annotations per class in full dataset' if SPLIT == 'full' else f"Number of annotations per class in {SPLIT} split",
        save_path=save_dir)    
    
    return class_counts

if __name__ == "__main__":
    class_counts = []
    for split in ["full", "train", "val", "test"]:
        class_counts.append(main(split))
    
    # Plot proportions
    for i in range(len(class_counts)):
        if i == 0: continue
        for j in class_counts[i].keys():
            class_counts[i][j] = f"{float(class_counts[i][j] / class_counts[0][j])*100.0:.2f}"

    for aux in class_counts:
        print(aux)
    