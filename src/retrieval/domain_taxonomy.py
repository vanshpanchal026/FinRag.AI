"""Comprehensive Indian Financial Taxonomy and Domain Query Expander for FinRAG-Eval.

Encodes complete domain intelligence across 15 financial categories:
1. Company Reports, Filings & Research Documents (Annual Report, P&L, BS, CF, Notes, BRSR, RPT)
2. Management Communication & Investor Interactions (Concall, Investor Deck, Guidance, CEO/CFO commentary)
3. Corporate Meetings, Announcements & Events (Board Meetings, AGM/EGM, Reg 30 disclosures, Ex-Dates)
4. Corporate Actions & Capital Restructuring (Dividends, Bonus, Splits, Buyback, Rights, QIP, Demerger)
5. IPO & Fundraising Terminology (DRHP, RHP, Anchor Investors, BRLM, Price Band, Lock-in)
6. Shareholding, Ownership & Institutional Activity (SHP, Promoter Pledges, FII/DII, SAST, Bulk/Block deals)
7. Business Developments & Catalysts (Order Book, Capex, Capacity Expansion, LOI/LOA, Approvals)
8. Financial Performance & Growth Metrics (Revenue, ASP, Gross Margin, EBITDA Margin, PAT, EPS, YoY, QoQ)
9. Returns, Balance-Sheet & Cash-Flow Metrics (ROE, ROCE, ROIC, Net Debt, DSCR, Working Capital Days, FCF)
10. Valuation Metrics & Equity-Research (P/E, Forward P/E, P/B, EV/EBITDA, DCF, WACC, SOTP, Margin of Safety)
11. Governance & Red-Flag Terminology (Auditor Qualifications, RPTs, Resignations, Pledges, Defaults)
12. Sector-Specific Business KPIs (Banks: NIM/NII/CASA/GNPA; IT: TCV/Attrition; FMCG: Volume/Price-mix; Pharma: USFDA 483/ANDA; Auto: ASP/Dealer Inventory; Telecom: ARPU)
13. Swing-Trading & Technicals (OHLC, RVOL, 52W High/Low, Breakout, SMA/EMA, RSI, MACD, VWAP)
14. Derivatives & Trading Risk (Open Interest/OI, PCR, IV, India VIX, Greeks, MWPL, F&O Ban, ASM/GSM)
15. Macro & Industry Events (RBI MPC, Repo Rate, CPI/WPI, GDP, PMI, Budget, FII/DII Flows)

Provides deterministic query expansion for BM25Okapi and Dense embeddings to maximize Recall@K.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


# Comprehensive financial acronyms, abbreviations, and canonical expansions
FINANCIAL_ACRONYMS: Dict[str, Dict[str, Any]] = {
    # 8. Financial Performance & Growth Metrics
    "ebitda": {
        "expansion": "Earnings Before Interest Taxes Depreciation and Amortisation",
        "synonyms": ["operating profit", "ebitda margin", "operating ebitda"],
        "category": "performance_metrics",
    },
    "ebit": {
        "expansion": "Earnings Before Interest and Taxes",
        "synonyms": ["operating profit", "pbit"],
        "category": "performance_metrics",
    },
    "pbt": {
        "expansion": "Profit Before Tax",
        "synonyms": ["earnings before tax", "ebt"],
        "category": "performance_metrics",
    },
    "pat": {
        "expansion": "Profit After Tax",
        "synonyms": ["net profit", "bottomline", "net income"],
        "category": "performance_metrics",
    },
    "eps": {
        "expansion": "Earnings Per Share",
        "synonyms": ["basic eps", "diluted eps"],
        "category": "performance_metrics",
    },
    "opm": {
        "expansion": "Operating Profit Margin",
        "synonyms": ["operating margin", "ebitda margin"],
        "category": "performance_metrics",
    },
    "npm": {
        "expansion": "Net Profit Margin",
        "synonyms": ["pat margin", "net margin"],
        "category": "performance_metrics",
    },
    "yoy": {
        "expansion": "Year-on-Year",
        "synonyms": ["annual growth", "yearly comparison"],
        "category": "growth_terms",
    },
    "qoq": {
        "expansion": "Quarter-on-Quarter",
        "synonyms": ["sequential growth", "quarterly comparison"],
        "category": "growth_terms",
    },
    "cagr": {
        "expansion": "Compound Annual Growth Rate",
        "synonyms": ["annualized growth rate"],
        "category": "growth_terms",
    },
    "ttm": {
        "expansion": "Trailing Twelve Months",
        "synonyms": ["last 12 months", "l12m"],
        "category": "growth_terms",
    },
    "asp": {
        "expansion": "Average Selling Price",
        "synonyms": ["realisation per unit", "average realisation"],
        "category": "operational_metrics",
    },

    # 9. Returns, Balance-Sheet Strength & Cash-Flow
    "roe": {
        "expansion": "Return on Equity",
        "synonyms": ["return on net worth", "ronw"],
        "category": "return_ratios",
    },
    "roce": {
        "expansion": "Return on Capital Employed",
        "synonyms": ["operating roce", "capital efficiency"],
        "category": "return_ratios",
    },
    "roic": {
        "expansion": "Return on Invested Capital",
        "synonyms": ["invested capital return"],
        "category": "return_ratios",
    },
    "roa": {
        "expansion": "Return on Assets",
        "synonyms": ["asset productivity"],
        "category": "return_ratios",
    },
    "cfo": {
        "expansion": "Cash Flow from Operations",
        "synonyms": ["operating cash flow", "ocf"],
        "category": "cash_flow",
    },
    "fcf": {
        "expansion": "Free Cash Flow",
        "synonyms": ["fcf yield", "free cash flow margin"],
        "category": "cash_flow",
    },
    "dscr": {
        "expansion": "Debt Service Coverage Ratio",
        "synonyms": ["debt service coverage", "debt servicing"],
        "category": "solvency_ratios",
    },
    "ccc": {
        "expansion": "Cash Conversion Cycle",
        "synonyms": ["working capital cycle", "net operating cycle"],
        "category": "working_capital",
    },
    "wc": {
        "expansion": "Working Capital",
        "synonyms": ["current assets less current liabilities", "net working capital"],
        "category": "working_capital",
    },
    "bvps": {
        "expansion": "Book Value Per Share",
        "synonyms": ["net asset value per share", "book value"],
        "category": "valuation_metrics",
    },

    # 10. Valuation Metrics & Equity Research
    "pe": {
        "expansion": "Price to Earnings Ratio",
        "synonyms": ["p/e ratio", "trailing p/e", "forward p/e", "multiple"],
        "category": "valuation_metrics",
    },
    "pb": {
        "expansion": "Price to Book Value Ratio",
        "synonyms": ["p/b ratio", "price to book"],
        "category": "valuation_metrics",
    },
    "ev": {
        "expansion": "Enterprise Value",
        "synonyms": ["firm value", "ev/ebitda", "ev/sales"],
        "category": "valuation_metrics",
    },
    "peg": {
        "expansion": "Price to Earnings to Growth Ratio",
        "synonyms": ["peg ratio"],
        "category": "valuation_metrics",
    },
    "dcf": {
        "expansion": "Discounted Cash Flow",
        "synonyms": ["intrinsic value", "dcf valuation", "terminal value"],
        "category": "valuation_metrics",
    },
    "wacc": {
        "expansion": "Weighted Average Cost of Capital",
        "synonyms": ["cost of capital", "hurdle rate"],
        "category": "valuation_metrics",
    },
    "sotp": {
        "expansion": "Sum of the Parts",
        "synonyms": ["sotp valuation", "segmental sum-of-the-parts"],
        "category": "valuation_metrics",
    },

    # 11. Governance & Red Flags
    "rpt": {
        "expansion": "Related Party Transactions",
        "synonyms": ["related party disclosures", "promoter transactions", "inter-corporate loans"],
        "category": "governance_red_flags",
    },

    # 12. Sector-Specific Business KPIs (Banks & NBFCs)
    "nim": {
        "expansion": "Net Interest Margin",
        "synonyms": ["interest margin", "spread"],
        "category": "banking_kpis",
    },
    "nii": {
        "expansion": "Net Interest Income",
        "synonyms": ["interest income minus interest expended"],
        "category": "banking_kpis",
    },
    "casa": {
        "expansion": "Current Account Savings Account",
        "synonyms": ["casa ratio", "low-cost deposits"],
        "category": "banking_kpis",
    },
    "gnpa": {
        "expansion": "Gross Non-Performing Assets",
        "synonyms": ["gross npa", "bad loans", "npa ratio"],
        "category": "banking_kpis",
    },
    "nnpa": {
        "expansion": "Net Non-Performing Assets",
        "synonyms": ["net npa", "net bad loans"],
        "category": "banking_kpis",
    },
    "pcr": {
        "expansion": "Provision Coverage Ratio",
        "synonyms": ["provisioning coverage", "npa coverage"],
        "category": "banking_kpis",
    },
    "cet1": {
        "expansion": "Common Equity Tier 1 Ratio",
        "synonyms": ["capital adequacy", "tier 1 capital", "crar"],
        "category": "banking_kpis",
    },
    "aum": {
        "expansion": "Assets Under Management",
        "synonyms": ["loan book", "assets managed", "portfolio scale"],
        "category": "nbfc_amc_kpis",
    },

    # Sector-Specific Business KPIs (IT Services)
    "tcv": {
        "expansion": "Total Contract Value",
        "synonyms": ["deal wins", "order intake", "contract wins"],
        "category": "it_services_kpis",
    },
    "acv": {
        "expansion": "Annual Contract Value",
        "synonyms": ["annual deal size"],
        "category": "it_services_kpis",
    },
    "cc": {
        "expansion": "Constant Currency",
        "synonyms": ["constant currency growth", "forex-neutral growth"],
        "category": "it_services_kpis",
    },

    # Sector-Specific Business KPIs (Insurance)
    "ape": {
        "expansion": "Annualized Premium Equivalent",
        "synonyms": ["weighted premium income", "new business premium"],
        "category": "insurance_kpis",
    },
    "vnb": {
        "expansion": "Value of New Business",
        "synonyms": ["vnb margin", "new business profitability"],
        "category": "insurance_kpis",
    },

    # Sector-Specific Business KPIs (Retail / QSR / Consumer)
    "sssg": {
        "expansion": "Same Store Sales Growth",
        "synonyms": ["like-for-like sales growth", "lfl growth"],
        "category": "retail_kpis",
    },

    # Sector-Specific Business KPIs (Pharma)
    "anda": {
        "expansion": "Abbreviated New Drug Application",
        "synonyms": ["generic drug approval", "usfda filings"],
        "category": "pharma_kpis",
    },

    # Sector-Specific Business KPIs (Telecom / Platforms)
    "arpu": {
        "expansion": "Average Revenue Per User",
        "synonyms": ["average revenue per subscriber", "monthly arpu"],
        "category": "telecom_platform_kpis",
    },
    "gmv": {
        "expansion": "Gross Merchandise Value",
        "synonyms": ["gross order value", "gov"],
        "category": "platform_kpis",
    },
    "arr": {
        "expansion": "Annual Recurring Revenue",
        "synonyms": ["annualized run rate", "subscription revenue"],
        "category": "saas_kpis",
    },
    "nrr": {
        "expansion": "Net Retention Rate",
        "synonyms": ["net dollar retention", "ndr"],
        "category": "saas_kpis",
    },

    # 4. Corporate Actions & Capital Restructuring
    "qip": {
        "expansion": "Qualified Institutions Placement",
        "synonyms": ["institutional share placement", "institutional allotment"],
        "category": "corporate_actions",
    },
    "esop": {
        "expansion": "Employee Stock Option Plan",
        "synonyms": ["stock options", "share allotment"],
        "category": "corporate_actions",
    },
    "ofs": {
        "expansion": "Offer for Sale",
        "synonyms": ["promoter stake sale", "secondary offering"],
        "category": "corporate_actions",
    },

    # 5. IPO & Fundraising Documents
    "drhp": {
        "expansion": "Draft Red Herring Prospectus",
        "synonyms": ["draft offer document", "sebi ipo filing"],
        "category": "ipo_filings",
    },
    "rhp": {
        "expansion": "Red Herring Prospectus",
        "synonyms": ["ipo prospectus", "offer document"],
        "category": "ipo_filings",
    },
    "brlm": {
        "expansion": "Book Running Lead Manager",
        "synonyms": ["merchant banker", "lead manager"],
        "category": "ipo_filings",
    },

    # 6. Shareholding & Ownership
    "shp": {
        "expansion": "Shareholding Pattern",
        "synonyms": ["promoter holding", "public shareholding", "ownership structure"],
        "category": "shareholding",
    },
    "fii": {
        "expansion": "Foreign Institutional Investors",
        "synonyms": ["fpi", "foreign portfolio investors"],
        "category": "shareholding",
    },
    "dii": {
        "expansion": "Domestic Institutional Investors",
        "synonyms": ["mutual fund holding", "insurance holding"],
        "category": "shareholding",
    },
    "sast": {
        "expansion": "Substantial Acquisition of Shares and Takeovers",
        "synonyms": ["takeover code disclosures", "insider acquisition"],
        "category": "shareholding",
    },

    # 7. Business Catalysts & Capex
    "loi": {
        "expansion": "Letter of Intent",
        "synonyms": ["preliminary contract", "agreement in principle"],
        "category": "business_catalysts",
    },
    "loa": {
        "expansion": "Letter of Award",
        "synonyms": ["contract award", "tender award"],
        "category": "business_catalysts",
    },
    "capex": {
        "expansion": "Capital Expenditure",
        "synonyms": ["capacity expansion", "plant commissioning", "capital outlay"],
        "category": "business_catalysts",
    },

    # 13. Swing-Trading & Technicals
    "ohlc": {
        "expansion": "Open High Low Close",
        "synonyms": ["price range", "candlestick"],
        "category": "trading_technical",
    },
    "rvol": {
        "expansion": "Relative Volume",
        "synonyms": ["volume surge", "unusual volume"],
        "category": "trading_technical",
    },
    "vwap": {
        "expansion": "Volume Weighted Average Price",
        "synonyms": ["anchored vwap", "benchmark price"],
        "category": "trading_technical",
    },
    "sma": {
        "expansion": "Simple Moving Average",
        "synonyms": ["moving average", "ma"],
        "category": "trading_technical",
    },
    "ema": {
        "expansion": "Exponential Moving Average",
        "synonyms": ["exponential average"],
        "category": "trading_technical",
    },
    "rsi": {
        "expansion": "Relative Strength Index",
        "synonyms": ["momentum indicator", "overbought oversold"],
        "category": "trading_technical",
    },
    "macd": {
        "expansion": "Moving Average Convergence Divergence",
        "synonyms": ["macd signal", "macd crossover"],
        "category": "trading_technical",
    },

    # 14. Derivatives & Trading Risk
    "oi": {
        "expansion": "Open Interest",
        "synonyms": ["futures open interest", "options open interest", "oi buildup"],
        "category": "derivatives",
    },
    "pcr": {
        "expansion": "Put Call Ratio",
        "synonyms": ["put-to-call ratio", "sentiment indicator"],
        "category": "derivatives",
    },
    "iv": {
        "expansion": "Implied Volatility",
        "synonyms": ["option volatility", "volatility surface"],
        "category": "derivatives",
    },
    "vix": {
        "expansion": "Volatility Index",
        "synonyms": ["india vix", "market volatility"],
        "category": "derivatives",
    },
    "mwpl": {
        "expansion": "Market Wide Position Limit",
        "synonyms": ["f&o ban limit", "position ceiling"],
        "category": "derivatives",
    },
    "asm": {
        "expansion": "Additional Surveillance Measure",
        "synonyms": ["surveillance framework", "sebi surveillance"],
        "category": "market_restrictions",
    },
    "gsm": {
        "expansion": "Graded Surveillance Measure",
        "synonyms": ["graded surveillance list", "microcap surveillance"],
        "category": "market_restrictions",
    },

    # 15. Macro & Industry Events
    "mpc": {
        "expansion": "Monetary Policy Committee",
        "synonyms": ["rbi monetary policy", "repo rate decision"],
        "category": "macro_events",
    },
    "cpi": {
        "expansion": "Consumer Price Index",
        "synonyms": ["retail inflation", "headline inflation"],
        "category": "macro_events",
    },
    "wpi": {
        "expansion": "Wholesale Price Index",
        "synonyms": ["wholesale inflation", "producer prices"],
        "category": "macro_events",
    },
    "pmi": {
        "expansion": "Purchasing Managers Index",
        "synonyms": ["manufacturing pmi", "services pmi"],
        "category": "macro_events",
    },
    "iip": {
        "expansion": "Index of Industrial Production",
        "synonyms": ["industrial output", "factory output"],
        "category": "macro_events",
    },
}

# The 15 Financial Taxonomy Categories
TAXONOMY_CATEGORIES = {
    1: "Company Reports, Filings & Research Documents",
    2: "Management Communication & Investor Interactions",
    3: "Corporate Meetings, Announcements & Events",
    4: "Corporate Actions & Capital Restructuring",
    5: "IPO & Fundraising Terminology",
    6: "Shareholding, Ownership & Institutional Activity",
    7: "Business Developments & Catalysts",
    8: "Financial Performance & Growth Metrics",
    9: "Returns, Balance-Sheet & Cash-Flow Metrics",
    10: "Valuation Metrics & Equity-Research Concepts",
    11: "Governance & Red-Flag Terminology",
    12: "Sector-Specific Business KPIs",
    13: "Swing-Trading, Price & Volume Technicals",
    14: "Derivatives, Market Restrictions & Trading Risk",
    15: "Macro & Industry-Level Events",
}


def expand_financial_query(query: str) -> Tuple[str, List[str], Optional[str]]:
    """Deterministically expand financial query terms for optimal dense and sparse retrieval.

    Args:
        query: Raw financial question submitted by user.

    Returns:
        Tuple of:
            - expanded_query (str): Query with full acronym expansions for Dense embeddings.
            - sparse_keywords (List[str]): High-signal keyword tokens for BM25Okapi search.
            - detected_category (Optional[str]): Detected taxonomy domain if identifiable.
    """
    clean_query = query.strip()
    words = re.findall(r"\b[A-Za-z0-9_%/+-]+\b", clean_query)
    
    additional_terms: List[str] = []
    sparse_keywords: Set[str] = set(w.lower() for w in words)
    detected_categories: Set[str] = set()

    for word in words:
        w_lower = word.lower()
        if w_lower in FINANCIAL_ACRONYMS:
            entry = FINANCIAL_ACRONYMS[w_lower]
            expansion = entry["expansion"]
            synonyms = entry.get("synonyms", [])
            cat = entry.get("category", "")
            
            if cat:
                detected_categories.add(cat)
            
            additional_terms.append(f"({expansion})")
            # Inject expansion words into sparse search
            for exp_word in re.findall(r"\b[A-Za-z0-9]+\b", expansion.lower()):
                if len(exp_word) > 2:
                    sparse_keywords.add(exp_word)
            for syn in synonyms:
                for syn_word in re.findall(r"\b[A-Za-z0-9]+\b", syn.lower()):
                    if len(syn_word) > 2:
                        sparse_keywords.add(syn_word)

    expanded_query = clean_query
    if additional_terms:
        expanded_query = f"{clean_query} {' '.join(additional_terms)}"

    primary_category = next(iter(detected_categories)) if detected_categories else None

    return expanded_query, sorted(list(sparse_keywords)), primary_category
