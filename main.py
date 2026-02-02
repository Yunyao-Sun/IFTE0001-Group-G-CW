import sys
import os
import glob # used to locate image files
from dotenv import load_dotenv
from crew import StockAnalysisCrew

load_dotenv(override=True) 

def save_markdown_report(report_text, ticker):
    """
    Save report and images as one Markdown file
    """
    output_dir = "analysis_output"
    filename = f"{ticker}_final_report.md"
    
    # 1. Prepare Markdown text for images
    # Find all PNG images related to the ticker
    image_files = glob.glob(f"{output_dir}/{ticker}_*.png")
    
    images_markdown = "\n\n## Visual Analysis Charts\n"
    if image_files:
        for img_path in image_files:
            # Generate Markdown image link: ![Title](path)
            img_name = os.path.basename(img_path)
            images_markdown += f"### {img_name}\n![{img_name}]({img_path})\n\n"
    else:
        images_markdown += "> No charts detected in output directory.\n"

    # 2. Merge Content
    full_content = str(report_text) + images_markdown

    # 3. Save files
    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_content)
    
    return filename

def run():
    print(f"DEBUG: Using Key: {os.getenv('OPENAI_API_KEY')[:10]}...")
    
    # Extract ticker as a variable for later use
    target_stock = 'BP' 
    
    inputs = {
        'query': 'Analyze the financial health and market performance.',
        'company_stock': target_stock,
    }
    
    # Get the output from Crew execution
    result = StockAnalysisCrew().crew().kickoff(inputs=inputs)
    
    # --- Generate consolidated report ---
    print("\n... Generating combined report ...")
    report_file = save_markdown_report(result, target_stock)
    print(f"✅ Report saved successfully: {report_file}")
    
    return result

def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'query': 'What is last years revenue',
        'company_stock': 'AMZN',
    }
    try:
        StockAnalysisCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

if __name__ == "__main__":
    print("## Welcome to Stock Analysis Crew")
    print('-------------------------------')
    result = run()
    print("\n\n########################")
    print("## Here is the Report")
    print("########################\n")
    print(result)