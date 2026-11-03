"""Benchmark generator for IndFinRAG-Bench.

Generates benchmarks/indfinrag_bench.json:
- 25 Indian stocks across large-cap, mid-cap, small-cap.
- 100 tricky questions across 4 stress-test categories (25 each):
  1. Unit Conversion Traps (Lakhs vs Crores vs Millions)
  2. Multi-Column Table & Scanned BSE Result Extraction
  3. Concall Guidance Revisions & Red-Flag Attribution
  4. Derived Margin & Growth Arithmetic (YoY %, QoQ %, bps)
"""

import json
from pathlib import Path

BENCHMARK_STOCKS = [
    # Large-Cap (10)
    ("RELIANCE", "Reliance Industries Ltd", "Oil & Gas / Retail / Telecom"),
    ("TCS", "Tata Consultancy Services Ltd", "IT Services"),
    ("HDFCBANK", "HDFC Bank Ltd", "Banking & Financial Services"),
    ("INFOSYS", "Infosys Ltd", "IT Consulting"),
    ("LT", "Larsen & Toubro Ltd", "Engineering & Infrastructure"),
    ("TITAN", "Titan Company Ltd", "Watches & Jewellery"),
    ("TRENT", "Trent Ltd", "Retail / Apparel"),
    ("ZOMATO", "Zomato Ltd", "Food Delivery / Quick Commerce"),
    ("SUNPHARMA", "Sun Pharmaceutical Industries Ltd", "Pharmaceuticals"),
    ("ADANIPORTS", "Adani Ports and Special Economic Zone Ltd", "Ports & Logistics"),
    # Mid-Cap (7)
    ("DIXON", "Dixon Technologies (India) Ltd", "EMS / Electronics Manufacturing"),
    ("TATAELXSI", "Tata Elxsi Ltd", "IT & Product Engineering"),
    ("COFORGE", "Coforge Ltd", "IT Solutions"),
    ("CDSL", "Central Depository Services (India) Ltd", "Capital Market Infrastructure"),
    ("BSELTD", "BSE Ltd", "Stock Exchange"),
    ("APOLLOHOSP", "Apollo Hospitals Enterprise Ltd", "Healthcare"),
    ("DIVISLAB", "Divi's Laboratories Ltd", "Pharma API"),
    # Small / High-Growth (8)
    ("KPIGREEN", "KPI Green Energy Ltd", "Renewable Energy / Solar"),
    ("SUZLON", "Suzlon Energy Ltd", "Wind Energy"),
    ("ALLCARGO", "Allcargo Logistics Ltd", "Logistics & Supply Chain"),
    ("STEELMAN", "Steelman Telecom Ltd", "Telecom Infrastructure"),
    ("KRISHNAFIL", "Krishna Filament Industries Ltd", "Industrial Manufacturing"),
    ("ITC", "ITC Ltd", "FMCG / Cigarettes / Hotels"),
    ("HAL", "Hindustan Aeronautics Ltd", "Aerospace & Defence"),
    ("KPITTECH", "KPIT Technologies Ltd", "Automotive Software"),
]

def build_benchmark():
    items = []
    
    # --------------------------------------------------------------------------
    # CATEGORY A: Unit Conversion Traps (Lakhs vs Crores vs Millions) - 25 items
    # --------------------------------------------------------------------------
    for idx, (sym, name, sector) in enumerate(BENCHMARK_STOCKS, start=1):
        lakhs_val = 50000 + (idx * 3450)
        cr_val = round(lakhs_val / 100.0, 2)
        p_no = 3 + (idx % 8)
        
        chunk_text = (
            f"(Rs. in Lakhs)\n"
            f"Statement of Standalone Financial Results for {name} for the period ended September 30, 2026\n"
            f"| Particulars | Quarter Ended 30-Sep-2026 | Quarter Ended 30-Jun-2026 |\n"
            f"|---|---|---|\n"
            f"| Total Income from Operations | {lakhs_val:,.2f} | {lakhs_val - 2500:,.2f} |\n"
            f"| Net Profit for the period (after tax) | {int(lakhs_val * 0.16):,d} | {int((lakhs_val - 2500) * 0.15):,d} |\n"
            f"The above results were reviewed by the Audit Committee and approved by the Board of Directors."
        )
        
        items.append({
            "query_id": f"indfin_catA_{idx:03d}",
            "company": sym,
            "company_name": name,
            "quarter": "Q2_FY27",
            "category": "unit_conversion_trap",
            "doc_type": "Financial_Results",
            "query": f"What was {sym}'s Total Income from Operations for the quarter ended 30-Sep-2026 expressed in Rs. Crores?",
            "golden_context": {
                "chunk_id": f"{sym}_Q2_FY27_p{p_no:02d}_c01",
                "source_file": f"{sym}_BSE_Financial_Results_Q2FY27.pdf",
                "page_no": p_no,
                "text": chunk_text
            },
            "ground_truth": {
                "source_filing_unit": "Lakhs",
                "source_filing_value": f"{lakhs_val:,.2f}",
                "normalized_crores": f"{cr_val:,.2f} Cr",
                "prior_period_value": f"{lakhs_val - 2500:,.2f}",
                "target_chunks": [f"{sym}_Q2_FY27_p{p_no:02d}_c01"],
                "expected_answer": f"Total Income from Operations for {sym} was {cr_val:,.2f} Crores ({lakhs_val:,.2f} Lakhs) [Source: {sym}_BSE_Financial_Results_Q2FY27.pdf | Page: {p_no} | Chunk: {sym}_Q2_FY27_p{p_no:02d}_c01].",
                "expected_figures": [f"{cr_val:,.2f}", f"{lakhs_val:,.2f}"],
                "source_unit": "Lakhs",
                "target_unit": "Crores",
                "unit_multiplier": 100.0,
                "naive_rag_trap": f"Naive baseline outputs {lakhs_val:,.2f} Crores instead of converting Lakhs to Crores, inflating figures 100x."
            }
        })

    # --------------------------------------------------------------------------
    # CATEGORY B: Multi-Column Table & Scanned BSE Result Extraction - 25 items
    # --------------------------------------------------------------------------
    for idx, (sym, name, sector) in enumerate(BENCHMARK_STOCKS, start=1):
        std_rev = 1200.0 + (idx * 85.5)
        con_rev = std_rev * 1.35
        std_pat = std_rev * 0.18
        con_pat = con_rev * 0.19
        p_no = 2 + (idx % 6)
        
        chunk_text = (
            f"(Rs. in Crores)\n"
            f"Segment Disclosures & Unaudited Financial Results for {name}\n"
            f"| Particulars | Standalone Q2 FY27 | Standalone Q1 FY27 | Consolidated Q2 FY27 | Consolidated Q1 FY27 |\n"
            f"|---|---|---|---|---|\n"
            f"| Revenue from Operations | {std_rev:,.1f} | {std_rev - 60:,.1f} | {con_rev:,.1f} | {con_rev - 90:,.1f} |\n"
            f"| Profit After Tax (PAT) | {std_pat:,.1f} | {std_pat - 10:,.1f} | {con_pat:,.1f} | {con_pat - 15:,.1f} |\n"
            f"Scanned table verification note: Consolidated includes international subsidiary revenues."
        )

        items.append({
            "query_id": f"indfin_catB_{idx:03d}",
            "company": sym,
            "company_name": name,
            "quarter": "Q2_FY27",
            "category": "multi_column_scanned_table",
            "doc_type": "Financial_Results",
            "query": f"What was {sym}'s Consolidated Revenue from Operations versus Standalone Revenue from Operations in Q2 FY27?",
            "golden_context": {
                "chunk_id": f"{sym}_Q2_FY27_p{p_no:02d}_c02",
                "source_file": f"{sym}_BSE_Financial_Results_Q2FY27.pdf",
                "page_no": p_no,
                "text": chunk_text
            },
            "ground_truth": {
                "target_chunks": [f"{sym}_Q2_FY27_p{p_no:02d}_c02"],
                "expected_answer": f"{sym} reported Consolidated Revenue from Operations of {con_rev:,.1f} Cr compared to Standalone Revenue of {std_rev:,.1f} Cr in Q2 FY27 [Source: {sym}_BSE_Financial_Results_Q2FY27.pdf | Page: {p_no} | Chunk: {sym}_Q2_FY27_p{p_no:02d}_c02].",
                "expected_figures": [f"{con_rev:,.1f}", f"{std_rev:,.1f}"],
                "standalone_metric": std_rev,
                "consolidated_metric": con_rev,
                "naive_rag_trap": "Naive sliding-window chunkers confuse the adjacent Standalone and Consolidated columns in scanned BSE tables."
            }
        })

    # --------------------------------------------------------------------------
    # CATEGORY C: Concall Guidance Revisions & Red-Flag Attribution - 25 items
    # --------------------------------------------------------------------------
    for idx, (sym, name, sector) in enumerate(BENCHMARK_STOCKS, start=1):
        old_guide = 18 + (idx % 6)
        new_guide = old_guide - 3
        capex_delay_months = 3 + (idx % 4)
        p_no = 8 + (idx % 10)
        
        chunk_text = (
            f"Earnings Conference Call Q&A Transcript - {name} (Q2 FY27)\n\n"
            f"Analyst (Morgan Stanley): Can management provide color on whether the previously communicated {old_guide}% volume growth guidance for FY27 remains on track?\n\n"
            f"Management: Given geopolitical headwinds and logistics bottlenecks in the Red Sea corridor, we are prudently tempering our full-year volume growth guidance to {new_guide}%. "
            f"Additionally, the commercial commissioning of our Phase-2 expansion is deferred by {capex_delay_months} months to Q4 FY27."
        )

        items.append({
            "query_id": f"indfin_catC_{idx:03d}",
            "company": sym,
            "company_name": name,
            "quarter": "Q2_FY27",
            "category": "guidance_revisions_red_flags",
            "doc_type": "Concall_Transcript",
            "query": f"Did {sym} management revise full-year volume growth guidance or capex timelines during the Q2 FY27 concall?",
            "golden_context": {
                "chunk_id": f"{sym}_Q2_FY27_p{p_no:02d}_c04",
                "source_file": f"{sym}_Q2_Concall_Transcript.pdf",
                "page_no": p_no,
                "text": chunk_text
            },
            "ground_truth": {
                "target_chunks": [f"{sym}_Q2_FY27_p{p_no:02d}_c04"],
                "expected_answer": f"Management revised full-year volume growth guidance downwards from {old_guide}% to {new_guide}% citing Red Sea bottlenecks, and deferred Phase-2 commissioning by {capex_delay_months} months to Q4 FY27 [Source: {sym}_Q2_Concall_Transcript.pdf | Page: {p_no} | Chunk: {sym}_Q2_FY27_p{p_no:02d}_c04].",
                "expected_figures": [f"{old_guide}%", f"{new_guide}%", f"{capex_delay_months}"],
                "previous_guidance": f"{old_guide}%",
                "revised_guidance": f"{new_guide}%",
                "naive_rag_trap": "Baseline models split the analyst question from the management answer, losing the negative revision context."
            }
        })

    # --------------------------------------------------------------------------
    # CATEGORY D: Derived Margin & Growth Arithmetic (YoY %, QoQ %, bps) - 25 items
    # --------------------------------------------------------------------------
    for idx, (sym, name, sector) in enumerate(BENCHMARK_STOCKS, start=1):
        q1_margin = 16.5 + (idx % 7) * 1.2
        q2_margin = q1_margin + 2.3
        bps_expansion = int(round((q2_margin - q1_margin) * 100))
        q1_rev = 1000.0 + (idx * 50.0)
        q2_rev = q1_rev * 1.085
        growth_pct = round(((q2_rev - q1_rev) / q1_rev) * 100.0, 2)
        p_no = 5 + (idx % 9)

        chunk_text = (
            f"(Rs. in Crores)\n"
            f"Investor Presentation - Financial Highlights for {name}\n"
            f"Revenue from Operations reached {q2_rev:,.2f} Cr in Q2 FY27 against {q1_rev:,.2f} Cr in Q1 FY27.\n"
            f"Operating EBITDA margin stood at {q2_margin:.1f}% in Q2 FY27 compared to {q1_margin:.1f}% in Q1 FY27."
        )

        items.append({
            "query_id": f"indfin_catD_{idx:03d}",
            "company": sym,
            "company_name": name,
            "quarter": "Q2_FY27",
            "category": "derived_margin_growth_ast",
            "doc_type": "Investor_Presentation",
            "query": f"What was {sym}'s QoQ Revenue growth rate and how many basis points (bps) did EBITDA margin expand in Q2 FY27?",
            "golden_context": {
                "chunk_id": f"{sym}_Q2_FY27_p{p_no:02d}_c03",
                "source_file": f"{sym}_Investor_Presentation_Q2FY27.pdf",
                "page_no": p_no,
                "text": chunk_text
            },
            "ground_truth": {
                "target_chunks": [f"{sym}_Q2_FY27_p{p_no:02d}_c03"],
                "derived_yoy_pct": f"+{growth_pct:.2f}%",
                "expected_answer": f"Revenue from Operations grew by {growth_pct:.2f}% QoQ (from {q1_rev:,.2f} Cr to {q2_rev:,.2f} Cr), while EBITDA margin expanded by {bps_expansion} bps from {q1_margin:.1f}% to {q2_margin:.1f}% [Source: {sym}_Investor_Presentation_Q2FY27.pdf | Page: {p_no} | Chunk: {sym}_Q2_FY27_p{p_no:02d}_c03].",
                "expected_figures": [f"{q2_rev:,.2f}", f"{q1_rev:,.2f}", f"{growth_pct:.2f}%", f"{bps_expansion}", f"{q2_margin:.1f}%", f"{q1_margin:.1f}%"],
                "derived_growth_pct": growth_pct,
                "derived_bps_expansion": bps_expansion,
                "ast_formula_growth": f"(({q2_rev:.2f} - {q1_rev:.2f}) / {q1_rev:.2f}) * 100",
                "ast_formula_bps": f"({q2_margin:.1f} - {q1_margin:.1f}) * 100",
                "naive_rag_trap": "Naive regex auditors flag the derived percentages and basis points as hallucinations because the derived numbers do not appear verbatim in the filing."
            }
        })

    return items

if __name__ == "__main__":
    benchmark_data = build_benchmark()
    benchmarks_dir = Path("benchmarks")
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    out_file1 = benchmarks_dir / "indfinrag_bench.json"
    out_file2 = benchmarks_dir / "ind_fin_rag_bench_100.json"

    with open(out_file1, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    with open(out_file2, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"Generated {len(benchmark_data)} benchmark items in {out_file1} and {out_file2}")
