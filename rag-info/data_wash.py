
import os
import re

# Configure the directory where the Markdown files are located
MD_OUTPUT_DIR = "./output"

# Define the paragraph pattern to match
pattern = re.compile(
    r"## Chapter 4 Conclusion\n\n"
    r"In this thesis we have presented a new method for the calculation.*?(?=\n\n|\Z)", 
    re.DOTALL
)

def clean_md_file(md_path):
    """Clean the target paragraph from the Markdown file"""
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find and remove the matching paragraph
    cleaned_content = re.sub(pattern, "", content)
    
    # Write the cleaned content back to the file
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(cleaned_content)
    print(f"Cleaned file: {md_path}")

def main():
    # Get all .md files in the directory
    md_files = [f for f in os.listdir(MD_OUTPUT_DIR) if f.endswith(".md")]
    
    # Clean each file
    for md_file in md_files:
        md_path = os.path.join(MD_OUTPUT_DIR, md_file)
        clean_md_file(md_path)

if __name__ == "__main__":
    main()