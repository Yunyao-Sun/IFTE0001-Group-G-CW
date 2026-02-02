import os
from dotenv import load_dotenv
from crew import StockAnalysisCrew
from main import save_markdown_report

def run_demo():
    load_dotenv(override=True)
    os.environ.pop("OPENAI_API_BASE", None)
    os.environ.pop("OPENAI_MODEL_NAME", None)

    print("🚀 Starting Demo: AI Stock Analysis for BP...")

    target_stock = 'BP' 
    inputs = {
        'query': 'Analyze the financial health and market performance.',
        'company_stock': target_stock,
    }

    try:
        print("\n🤖 Agents are working on the analysis (this may take a minute)...")
        result = StockAnalysisCrew().crew().kickoff(inputs=inputs)

        print("\n📊 Generating combined Markdown report with charts...")
        report_file = save_markdown_report(result, target_stock)
        
        print(f"\n✅ Demo Completed Successfully!")
        print(f"📂 Final Report: {os.path.abspath(report_file)}")
        print(f"📈 Charts saved in: {os.path.abspath('analysis_output/')}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    run_demo()
