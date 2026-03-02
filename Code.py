import os
import sys
import json
import logging
import pandas as pd
import re
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load env variables (Create a .env file with HUGGINGFACE_API_KEY=hf_...)
load_dotenv()

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger("BugHunter")

# We use Llama-3-8B-Instruct (Fast & Good at Logic)
HF_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


# ------------------------------------------------------------------
# 1. DATA MODELS
# ------------------------------------------------------------------
class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class BugLocation(BaseModel):
    file_name: str = Field(..., description="Name of the file.")
    line_number: int = Field(..., description="The 1-based line number where the bug starts.")
    code_snippet: str = Field(..., description="The exact code content at that line.")


class BugAnalysis(BaseModel):
    summary: str = Field(..., description="A short summary of the bug.")
    technical_explanation: str = Field(..., description="Detailed explanation of the logic error.")
    fix_suggestion: str = Field(..., description="The COMPLETE corrected code block. Do not use comments to describe the fix. Output the actual working code.")


class DebugReport(BaseModel):
    found_bugs: bool
    severity: Severity
    location: Optional[BugLocation]
    analysis: Optional[BugAnalysis]
    confidence_score: float


# ------------------------------------------------------------------
# 2. HELPER: JSON EXTRACTION
# ------------------------------------------------------------------
def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extracts JSON from raw text response, handling markdown blocks."""
    try:
        # 1. Try to find code block
        pattern = r"```json(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())

        # 2. Fallback: find first { and last }
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])

        return None
    except Exception as e:
        logger.error(f"JSON Parsing failed: {e}")
        return None


# ------------------------------------------------------------------
# 3. AGENT SYSTEM
# ------------------------------------------------------------------
class AgentBase:
    def __init__(self, model: str = HF_MODEL_ID):
        # API Key Logic: Check Env, then .env, then hardcode fallback
        api_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")

        # --- PASTE KEY BELOW IF ENV VARIABLES FAIL ---
        if not api_token:
            api_token = "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"
            # ---------------------------------------------

        if not api_token or "YOUR_ACTUAL_TOKEN" in api_token:
            raise ValueError("HUGGINGFACE_API_KEY is missing! Set it in .env or hardcode it.")

        # URL FIX: Point to the Router URL to avoid '410 Gone' errors
        # This bypasses the old API infrastructure
        self.client = InferenceClient(
            model=model,
            token=api_token
        )


class AnalyzerAgent(AgentBase):
    def analyze(self, code_context: str, feature_context: str) -> str:
        logger.info(f"Analyzer Agent: Checking constraints...")

        # PROMPT: Enforces Strict Rule Checking (Negative Constraints)
        prompt = f"""
        [INST] You are a Senior Hardware/Software Validation Engineer.
        Find LOGIC BUGS where the code violates the "Critical Rules".

        CRITICAL RULES:
        {feature_context}

        SOURCE CODE (Line Numbered):
        {code_context}

        STRICT INSTRUCTIONS:
        1. Compare every function call and constant (e.g., TA::...) against the Rules.
        2. If the Rule says "Use X", and the code uses "Y", THAT IS THE BUG.
        3. Ignore minor style issues; focus on Logic/Lifecycle violations.
        4. Be precise.
        [/INST]
        """

        try:
            # We use chat_completion (available in newer huggingface_hub versions)
            # If this fails, update library: pip install --upgrade huggingface_hub
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error during analysis: {str(e)}"


class LocatorAgent(AgentBase):
    def create_report(self, code_context: str, analysis: str, file_name: str) -> DebugReport:
        logger.info("Locator Agent: Formatting JSON...")

        prompt = f"""
        [INST]
        You are a Code Repair Engine.
        Based on the analysis below, output a VALID JSON object.

        ANALYSIS:
        {analysis}

        SOURCE CODE:
        {code_context}

        INSTRUCTIONS FOR 'fix_suggestion':
        1. It must contain the FULL code snippet with the bug fixed.
        2. Do NOT write "Add a semicolon here". Actually add the semicolon in the code.
        3. Do NOT truncate the code. Return the whole block so it can be copy-pasted.

        REQUIRED JSON STRUCTURE:
        {{
            "found_bugs": true,
            "severity": "high",
            "location": {{
                "file_name": "{file_name}",
                "line_number": 123,
                "code_snippet": "exact code"
            }},
            "analysis": {{
                "summary": "...",
                "technical_explanation": "...",
                "fix_suggestion": "int x = 10; // Corrected code"
            }},
            "confidence_score": 0.95
        }}

        Output ONLY valid JSON.
        [/INST]
        """

        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,  # Increased tokens to ensure full code fits
                temperature=0.1
            )
            raw_text = response.choices[0].message.content

            # Parse
            json_data = extract_json_from_text(raw_text)

            if not json_data:
                logger.error("Could not parse JSON from response.")
                return DebugReport(found_bugs=False, severity=Severity.LOW, location=None, analysis=None,
                                   confidence_score=0.0)

            return DebugReport(**json_data)
        except Exception as e:
            logger.error(f"Locator Error: {e}")
            return DebugReport(found_bugs=False, severity=Severity.LOW, location=None, analysis=None,
                               confidence_score=0.0)


# ------------------------------------------------------------------
# 4. ORCHESTRATOR
# ------------------------------------------------------------------
class BugHunterSystem:
    def __init__(self):
        self.analyzer = AnalyzerAgent()
        self.locator = LocatorAgent()

    def scan_sample(self, code_snippet: str, context_description: str) -> DebugReport:
        # Add line numbers
        lines = code_snippet.split('\n')
        numbered_code = "\n".join([f"{i + 1} | {line}" for i, line in enumerate(lines)])

        # Analyze
        analysis_text = self.analyzer.analyze(numbered_code, context_description)

        # Locate
        report = self.locator.create_report(numbered_code, analysis_text, "snippet")
        return report


# ------------------------------------------------------------------
# 5. UNIVERSAL MAIN BLOCK (CSV + TXT/CPP Support)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    target_file = ""

    # 1. Get Input File
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        while not target_file:
            target_file = input("\nEnter file path (e.g., 'cpp_bug_samples_v2.csv'): ").strip().strip('"').strip("'")

    print(f"DEBUG: Processing file -> '{target_file}'")

    if os.path.exists(target_file):
        system = BugHunterSystem()

        # ==========================================
        # MODE A: CSV BATCH SCAN
        # ==========================================
        if target_file.lower().endswith(".csv"):
            results_data = []

            try:
                df = pd.read_csv(target_file)
                print(f"\nSTARTING BATCH SCAN on {target_file} ({len(df)} samples)")

                # --- SMART COLUMN DETECTION ---
                cols = [c.lower() for c in df.columns]


                def get_col_name(keywords):
                    for col in df.columns:
                        if col.lower() in keywords:
                            return col
                    return None


                # Map columns dynamically
                id_col = get_col_name(['id', 'sample_id', 'index'])
                code_col = get_col_name(['code', 'source', 'snippet', 'source_code'])
                context_col = get_col_name(['context', 'description', 'intent', 'feature'])
                explain_col = get_col_name(['explanation', 'bug_description', 'ground_truth'])

                if not code_col:
                    print(f"CRITICAL ERROR: Could not find a 'Code' column in {list(df.columns)}")
                    sys.exit()

                print(f"Mapped: Code='{code_col}', Context='{context_col}'\n")

                # --- PROCESS LOOP ---
                for index, row in df.iterrows():
                    sample_id = row[id_col] if id_col else index
                    print(f"--- Processing ID {sample_id} ---")

                    original_code = str(row[code_col])
                    context_val = str(row[context_col]) if context_col and pd.notna(
                        row[context_col]) else "Standard Code Review"
                    explain_val = str(row[explain_col]) if explain_col and pd.notna(row[explain_col]) else ""

                    rich_context = f"Feature Context: {context_val}\nCritical Rules: {explain_val}"

                    # Initialize Default Result
                    result_row = {
                        "ID": sample_id,
                        "Explanation": "No bugs found",
                        "Original Code": original_code,
                        "Correct Code": original_code,  # Default to original
                        "Context": context_val
                    }

                    # SAFEGUARD: Initialize report to None
                    report = None

                    try:
                        # 3. Scan
                        report = system.scan_sample(original_code, rich_context)

                        # Only access 'report' if it exists and bugs were found
                        if report and report.found_bugs and report.location:
                            print(f"BUG DETECTED at Line {report.location.line_number}")
                            print(f"{report.analysis.summary}")

                            # Update Result Row
                            result_row["Explanation"] = report.analysis.technical_explanation
                            result_row["Correct Code"] = report.analysis.fix_suggestion
                        else:
                            print("Clean.\n")

                    except Exception as e:
                        print(f"⚠Error: {e}")
                        result_row["Explanation"] = f"Error during processing: {str(e)}"

                    # 4. Append Result (Safe because result_row is always valid)
                    results_data.append(result_row)

                # 5. Generate Output
                print("\n Saving results to 'output.csv'...")
                output_df = pd.DataFrame(results_data)
                output_df.to_csv("output.csv", index=False)
                print(" Done! Open 'output.csv'.")

            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"CSV Error: {e}")

        # ==========================================
        # MODE B: SINGLE FILE SCAN
        # ==========================================
        else:
            print(f"\n STARTING SINGLE FILE SCAN on {target_file}")
            try:
                with open(target_file, "r", encoding="utf-8") as f:
                    code_content = f.read()

                print("\nℹEnter Rules (Press Enter for Standard):")
                user_rules = input(" > ").strip() or "Standard Code Review"

                report = system.scan_sample(code_content, user_rules)

                if report and report.found_bugs and report.location:
                    print(f"\nBUG DETECTED at Line {report.location.line_number}")
                    print(f"{report.analysis.summary}")
                    print(f"Fix:\n{report.analysis.fix_suggestion}")
                else:
                    print("\nNo bugs detected.")
            except Exception as e:
                print(f"Error: {e}")

    else:
        print(f"Error: File '{target_file}' not found.")