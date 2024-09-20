import os
import sys
from pathlib import Path


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



def print_filtered_files(root_dir, excluded_words):
    """
    Searches for .py and .yaml files within root_dir and its subdirectories,
    excluding any paths that contain any of the excluded_words.
    For each qualifying file, prints the file path enclosed within separator lines
    followed by the file's content.

    Args:
        root_dir (str or Path): The root directory to search.
        excluded_words (list of str): List of substrings to exclude in file paths.
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: The root directory '{root_dir}' does not exist.")
        sys.exit(1)
    if not root_path.is_dir():
        print(f"Error: The path '{root_dir}' is not a directory.")
        sys.exit(1)

    # Define the file patterns to search for
    patterns = ('*.py', '*.yaml')

    # Iterate over each pattern
    for pattern in patterns:
        # Use rglob to recursively search for files matching the pattern
        for filepath in root_path.rglob(pattern):
            filepath_str = str(filepath)

            # Check if any of the excluded words are in the file path
            if any(excluded_word in filepath_str for excluded_word in excluded_words):
                continue  # Skip this file

            # Print the separator and file path
            separator = "=" * 40
            print(separator)
            print(filepath_str)
            print(separator)

            try:
                # Read and print the file content
                with filepath.open('r', encoding='utf-8') as file:
                    content = file.read()
                    print(content)
            except Exception as e:
                print(f"Error reading file '{filepath_str}': {e}")
            print()  # Add an empty line for better readability

if __name__ == "__main__":
    root_dir = "/home/admin2/Documents/repos/cct"
    excluded_dirs = ['.EXCLUDED', '_knowledge']
    excluded_words = ['_knowledge', '.EXCLUDED', 'flowers', 'tests', 'print_repo']

    print_tree(root_dir, excluded_dirs=excluded_dirs)
    print_filtered_files(root_dir, excluded_words)

