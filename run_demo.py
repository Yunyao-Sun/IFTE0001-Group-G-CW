import os
from dotenv import load_dotenv
from crew import StockAnalysisCrew
from main import save_markdown_report

def run_demo():
    # 1. 强制加载环境并清理可能存在的干扰变量
    load_dotenv(override=True)
    os.environ.pop("OPENAI_API_BASE", None)
    os.environ.pop("OPENAI_MODEL_NAME", None)

    print("🚀 Starting Demo: AI Stock Analysis for BP...")
    print(f"🔑 Using API Key (prefix): {os.getenv('OPENAI_API_KEY')[:10]}...")

    # 2. 设置演示目标
    target_stock = 'BP' 
    inputs = {
        'query': 'Analyze the financial health and market performance.',
        'company_stock': target_stock,
    }

    try:
        # 3. 运行 CrewAI 工作流
        print("\n🤖 Agents are working on the analysis (this may take a minute)...")
        result = StockAnalysisCrew().crew().kickoff(inputs=inputs)

        # 4. 生成并保存最终合并报告
        print("\n📊 Generating combined Markdown report with charts...")
        report_file = save_markdown_report(result, target_stock)
        
        print(f"\n✅ Demo Completed Successfully!")
        print(f"📂 Final Report: {os.path.abspath(report_file)}")
        print(f"📈 Charts saved in: {os.path.abspath('analysis_output/')}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    run_demo()