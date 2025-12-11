
import os
import json
import jinja2
import datetime
from dotenv import load_dotenv
from groq import Groq
from search import load_data, search_solutions

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY not found in .env file.")
    exit(1)

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

RELEVE_PATH = "releve.json"
REPORT_TEMPLATE_PATH = "report_template.html"
OUTPUT_REPORT_PATH = "report.html"

def summarize_releve(releve_path):
    """Summarizes the large releve.json file for the LLM."""
    print("Loading releve.json...")
    with open(releve_path, 'r') as f:
        data = json.load(f)
    
    # Calculate some aggregate stats to keep the prompt small
    total_boxes = len(data)
    if total_boxes == 0:
        return "No data found."
    
    # Just taking a sample of high dB events to show the "faiblesses"
    high_db_events = [d for d in data if d.get('LAeq_rating') in ['D', 'E']]
    
    summary = f"Total measurement points: {total_boxes}\n"
    summary += f"High noise rating (D/E) count: {len(high_db_events)}\n"
    
    # Add a few examples of the worst offenders
    sorted_events = sorted(data, key=lambda x: x.get('LAeq_segment_dB', 0), reverse=True)
    top_5 = sorted_events[:5]
    
    summary += "Top 5 Loudest Events:\n"
    for event in top_5:
        summary += f"- Time: {event.get('timestamp')}, dB: {event.get('LAeq_segment_dB')}, Rating: {event.get('LAeq_rating')}, Likely Sources: {', '.join(event.get('top_5_labels', []))}\n"
        
    return summary

def analyze_with_groq(summary):
    """Sends the summary to Groq to identify structural weaknesses."""
    print("Analyzing data with Groq (gemma2-9b-it)...")
    
    prompt = f"""
    You are an acoustics expert. Analyze the following summary of noise measurements from an apartment.
    Identify the likely "faiblesses structurelles phoniques" (structural acoustic weaknesses) in French.
    Focus on physical building elements (windows, walls, doors, floors, ceilings) that might be failing based on the noise sources (e.g., if 'Vehicle' noise is high, windows might be the issue; if 'Speech' from neighbors, walls/partitions).
    
    Data Summary:
    {summary}
    
    Provide a concise summary of the structural weaknesses in French. Do not offer solutions yet, just the diagnosis.
    """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        max_tokens=1024,
    )
    
    return chat_completion.choices[0].message.content

def generate_report(analysis, budget, recommended_solutions):
    """Generates the HTML report."""
    print("Generating HTML report...")
    
    template_loader = jinja2.FileSystemLoader(searchpath="./")
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template(REPORT_TEMPLATE_PATH)
    
    html_output = template.render(
        analysis=analysis,
        budget=budget,
        recommended_solutions=recommended_solutions,
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html_output)
    
    print(f"Report saved to {os.path.abspath(OUTPUT_REPORT_PATH)}")

def main():
    # 1. Summarize and Analyze
    summary = summarize_releve(RELEVE_PATH)
    analysis = analyze_with_groq(summary)
    
    print("\n--- Diagnostic IA ---")
    print(analysis)
    print("---------------------\n")
    
    # 2. Get Budget
    while True:
        try:
            budget_input = input("Entrez votre budget pour les travaux (en Euros): ")
            budget = float(budget_input)
            break
        except ValueError:
            print("Veuillez entrer un nombre valide.")
            
    # 3. Search Solutions
    # We use the analysis text as the query to find relevant solutions
    docs, vectorizer, tfidf_matrix = load_data()
    
    # We can split the analysis into sentences or use the whole text. 
    # Using the whole analysis text might be too broad, so let's try to extract key phrases or just use the whole block.
    # For simplicity, we search using the entire analysis text as a query.
    print(f"Searching for solutions based on diagnosis...")
    results = search_solutions(analysis, docs, vectorizer, tfidf_matrix, top_k=5)
    
    # 4. Generate Report
    # The template logic handles filtering by budget for display purposes (showing "Hors budget" vs "Compatible")
    # We pass the results structure expected by the template: list of objects with 'score' and 'doc' (which has 'solutions')
    
    # Transform results to match template expectation (list of dicts or objects)
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "doc": doc,
            "score": score
        })
        
    generate_report(analysis, budget, formatted_results)

if __name__ == "__main__":
    main()
