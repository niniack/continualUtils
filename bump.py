import re
import subprocess
import sys


def main():
    # Get PR title from command line argument
    pr_title = sys.argv[1]

    # Extract version using regex
    match = re.search(r"tag:\s*([\w.-]+)", pr_title)
    if not match:
        print("No tag found in PR title!")
        sys.exit(1)

    tag = match.group(1)

    # Tag the latest commit
    subprocess.run(["git", "tag", tag], check=False)

    # Update files with correct version
    subprocess.run(["poetry", "run", "dynamic-versioning"], check=True)


if __name__ == "__main__":
    main()
