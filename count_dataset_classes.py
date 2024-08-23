import os
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

def plot_bar_chart(class_counts, save_path: str = "plot.png"):
    categories = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(categories, counts, color='skyblue')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Class Counts')
    plt.xticks(rotation=45)
    plt.ylim(0, 2400)
    plt.tight_layout()
    # plt.show()

    # dave_path = os.path.join(save_path, relative_path)
    plt.savefig(save_path)

if __name__ == "__main__":

    SPLIT = "train"
    # Get the current directory where the Python script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Append a relative path to the current directory
    relative_path = f'datasets/trash_detection_split/labels/{SPLIT}'  # Replace with your relative path
    relative_path = f'datasets/trash_detection/labels'
    full_path = os.path.join(current_dir, relative_path)
    print(f"Full path: {full_path}")

    class_counts = read_txt_files(full_path)
    print(class_counts)
    
    save_dir = os.path.join(current_dir, "my_dataset_class_count_" + SPLIT + ".png")
    plot_bar_chart(class_counts, save_dir)    
