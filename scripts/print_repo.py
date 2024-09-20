import os

# List of directories to be excluded
excluded_dirs = ['.EXCLUDED', '_knowledge']

def print_tree(startpath, indent=0, excluded_dirs=[]):
    """
    Recursively prints a tree of files and directories starting from startpath.
    """
    try:
        files_and_dirs = os.listdir(startpath)
    except PermissionError:
        print(" " * indent + f"PermissionError: Cannot access '{startpath}'")
        return

    files_and_dirs.sort()
    num_items = len(files_and_dirs)

    file_counts = {}
    for name in files_and_dirs:
        if name.startswith('.') or name in excluded_dirs:
            continue

        path = os.path.join(startpath, name)
        file_ext = os.path.splitext(name)[1]

        if os.path.isfile(path) and file_ext != ".py":
            file_counts[file_ext] = file_counts.get(file_ext, 0) + 1

    printed_files = {ext: 0 for ext in file_counts}

    for i, name in enumerate(files_and_dirs):
        if name.startswith('.') or name in excluded_dirs:
            continue

        path = os.path.join(startpath, name)
        prefix = "├── " if i < num_items - 1 else "└── "

        if os.path.isdir(path):
            print(" " * indent + prefix + name)
            print_tree(path, indent + 4, excluded_dirs)
        else:
            file_ext = os.path.splitext(name)[1]
            if file_ext != ".py" and printed_files[file_ext] >= 5:
                continue

            print(" " * indent + prefix + name)
            if file_ext != ".py":
                printed_files[file_ext] += 1


def print_file_contents(startpath, excluded_dirs=[]):
    """
    Recursively prints the contents of all .py and .yaml files starting from startpath.
    """
    for root, dirs, files in os.walk(startpath):
        # Skip hidden directories and excluded directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in excluded_dirs]

        for file in files:
            if file.endswith(".py") or file.endswith(".yaml"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        print(f"\n{'=' * 40}\n{file_path}\n{'=' * 40}")
                        print(f.read())
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")


if __name__ == "__main__":
    # Specify the directory you want to print the tree for
    root_dir = "/home/admin2/Documents/repos/cct"
    print_tree(root_dir, excluded_dirs=excluded_dirs)
    print_file_contents(root_dir, excluded_dirs=excluded_dirs)
    """
    FlowersDataset
    test_ShapeClassificationNet
    """
