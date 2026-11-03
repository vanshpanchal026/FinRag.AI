import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

convs = [
    ("dbf8d7e3-f03f-47fd-a012-bac08c800c1f", "Institutional FinTech Frontend Architecture Setup"),
    ("e08c252f-4d87-4e51-87dc-758f20d3af88", "FinRAG-Eval Enterprise Scaling Plan"),
    ("0f284202-72b8-47b0-95cb-7c61445943ed", "Execute Financial RAG Plan"),
    ("6bd6806b-c21d-4802-b05e-b90c692468ef", "Understanding Project Architecture with Archify"),
    ("1ba7db20-24c3-4ac8-a0fe-889ee940359b", "Fixing PowerShell Virtual Environment Activation Error"),
    ("cc75aebc-cabf-42f7-9eb7-981cf88efe0a", "FinRAG-Eval Phase 1 Implementation")
]

base = Path(r"C:\Users\pc\.gemini\antigravity\brain")

for cid, title in convs:
    print(f"\n========================================================")
    print(f"CONVERSATION: {title}")
    print(f"ID: {cid}")
    print(f"========================================================")
    
    tpath = base / cid / ".system_generated" / "logs" / "transcript.jsonl"
    if not tpath.exists():
        print("Transcript does not exist!")
        continue
        
    user_inputs = []
    planner_steps = []
    
    with open(tpath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            stype = entry.get("type")
            idx = entry.get("step_index")
            content = entry.get("content", "")
            
            if stype == "USER_INPUT":
                user_inputs.append((idx, content))
            elif stype == "PLANNER_RESPONSE":
                # Check tool calls or text content
                tool_calls = entry.get("tool_calls", [])
                planner_steps.append((idx, content, [tc.get("toolSummary", "") for tc in tool_calls]))
                
    print(f"Total user inputs: {len(user_inputs)}")
    for idx, u in user_inputs:
        print(f"\n--- USER INPUT (step {idx}) ---")
        print(u[:400] + ("..." if len(u) > 400 else ""))
        
    print(f"\nTotal planner responses: {len(planner_steps)}")
    # Print the first, middle, and last planner responses with their tool calls
    if planner_steps:
        for p_idx, p_content, p_tools in planner_steps[-3:]:
            print(f"\n--- PLANNER STEP {p_idx} (tools: {p_tools}) ---")
            if p_content:
                print(p_content[:400] + ("..." if len(p_content) > 400 else ""))
