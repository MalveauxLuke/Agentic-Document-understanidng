============================================================
PROMPTS — USE THE PAPER'S AGENT PROMPTS
============================================================

Implement sleuth/agents/prompts.py.

The prompts must closely follow the prompts provided in Appendix F of the SLEUTH paper.

Important:
- Preserve the paper's intent and wording as much as possible.
- However, for implementation reliability, convert outputs to strict JSON where needed.
- Keep the original paper-style prompt text in comments or docstrings so we can trace the implementation back to the paper.
- Each prompt builder must accept optional instruction_md_text.
- If instruction_md_text is provided, include it near the top of every prompt under:

SOL-SPECIFIC INSTRUCTIONS:
{instruction_md_text}

You must follow these instructions exactly unless they conflict with system safety, code execution constraints, or the explicit task structure.

Implement:

format_sol_instruction_block(instruction_md_text: str | None) -> str

If instruction_md_text is None or empty, return an empty string.
Otherwise return the SOL instruction block.

============================================================
CLUE DISCOVERY AGENT PROMPT
============================================================

Implement:

build_clue_discovery_prompt(
    question: str,
    page_index: int,
    page_text: str | None = None,
    instruction_md_text: str | None = None
) -> str

Use this prompt content:

You are a Detective, an expert evidence collector for document question answering. Your task is to carefully examine the given PDF page and extract ALL evidence that might be relevant to answering the question.

Question: {question}

Page Information:
• Page Number: {page_num}

Your Task:
1. Carefully examine the page image!
2. Identify ALL facts, data points, and information that could help answer the question.
3. Extract specific evidence with:
• Exact quotes or data values
• Context where information appears
• Explanation of why it’s relevant

Output Format: Provide your analysis in the following JSON format:

{
  "page_number": {page_num},
  "has_relevant_evidence": true,
  "evidence_items": [
    {
      "evidence_type": "text/chart/table/figure",
      "content": "The actual evidence (quote, data, or description)",
      "location": "Description of where this appears on the page",
      "relevance": "Explanation of why this is relevant to answering the question",
      "confidence": "high/medium/low"
    }
  ],
  "page_summary": "Overall summary of findings from this page",
  "key_insights": "Any important insights or patterns noticed"
}

Important Guidelines:
• Be thorough - collect ALL potentially relevant evidence.
• Include exact numbers, percentages, and specific facts.
• Note relationships between data points.
• If the page is not relevant, explain why.
• Please think carefully and avoid generating content that does not conform to reality.
• Return valid JSON only. Do not wrap it in markdown fences.
• Do not invent evidence that is not visible on the page.
• If no relevant evidence exists, set "has_relevant_evidence" to false and use an empty evidence_items list.

Now examine the page and provide your evidence collection in valid JSON format.

If page_text is available, append:

Additional extracted page text for reference:
{page_text}

But make clear that the page image remains the primary evidence source.

Expected output:
- valid JSON
- no markdown
- no extra explanation outside JSON

============================================================
PAGE SCREENING AGENT PROMPT
============================================================

Implement:

build_page_screening_prompt(
    question: str,
    page_index: int,
    instruction_md_text: str | None = None
) -> str

Use this prompt content:

You are an expert at analyzing document pages and identifying relevant charts/figures/tables for answering questions.

Your task is to examine this PDF page image and determine:
1. Whether there are any charts, figures, tables, or diagrams on this page.
2. If charts/figures/tables exist, whether they are relevant to answering the given question.

Question: {question}

Page Number: {page_num}

Instructions:
1. First, carefully examine the page image to identify any visual elements like:
• Charts (bar charts, line charts, pie charts, etc.)
• Figures (diagrams, illustrations, photos, etc.)
• Tables (data tables, comparison tables, etc.)
• Infographics or other data visualizations

2. If you find charts/figures/tables, assess their relevance to the question:
• Completely Relevant: Directly contains information needed to answer the question.
• Relevant: Might contain related information, but relevance is uncertain.
• Irrelevant: The chart/figure/table exists but is clearly unrelated to the question.

3. If there are NO charts/figures/tables on this page and it is only pure text, mark relevance as "None".

The original paper used this free-form output format:

Has Chart: [Yes/No]
Relevance: [Completely Relevant/ Relevant/ Irrelevant]
Reasoning: [Brief explanation of your judgment, 1-2 sentences]

For this implementation, return strict JSON instead:

{
  "has_visual_element": true,
  "relevance": "Completely Relevant",
  "reasoning": "Brief explanation of your judgment, 1-2 sentences.",
  "keep_page": true
}

Allowed values:
- has_visual_element: true or false
- relevance: "Completely Relevant", "Relevant", "Irrelevant", or "None"
- keep_page: true only if relevance is "Completely Relevant" or "Relevant"; otherwise false

Now, analyze the provided page image and respond in valid JSON only.

Important:
• Do not wrap the JSON in markdown.
• Do not output extra text.
• Do not hallucinate visual elements.
• If the page has only pure text and no charts/figures/tables/diagrams, use:
  "has_visual_element": false,
  "relevance": "None",
  "keep_page": false

============================================================
DIFFICULTY ASSESSMENT AGENT PROMPT
============================================================

Implement:

build_difficulty_prompt(
    question: str,
    evidence_summary: str,
    instruction_md_text: str | None = None
) -> str

Use this prompt content:

You are an expert whose task is to evaluate the user’s query and any structured multimodal context to determine the optimal reasoning strategy.

Input:
• Question (Q): {question}
• Structured Context (C):
{evidence_summary}

Instructions: Analyze the query and context to determine the difficulty level d ∈ {0, 1} and generate a corresponding instruction set Γd.

1. Determine Difficulty Level (d):

• Mode 0 (Ordinary Mode, d = 0):
Select this if the question can be answered by direct lookup or simple extraction from the provided context.

• Mode 1 (Reasoning Mode, d = 1):
Select this if the question requires:
– Cross-page aggregation (combining clues from multiple pages).
– Numerical computation (summation, percentages, ratio calculations).
– Trend comparison (inferring information not explicitly stated).
– Multi-step inference (deducing implicit information).

2. Generate Instruction Set (Γd):
Create specific, actionable instructions to guide the Core Decision Agent.

Example for d = 1:
"Requires summing values from Page 2 (Table 1) and Page 5 (Text). Calculate the percentage growth."

Output Format (Strict JSON):

{
  "difficulty_level": 0,
  "instruction_set": "Specific reasoning instructions Γd for the next agent."
}

Important:
• Return valid JSON only.
• Do not wrap the JSON in markdown.
• Do not output extra text.
• difficulty_level must be either 0 or 1.
• If unsure, choose d = 1 only when the evidence clearly requires multi-step reasoning, calculation, cross-page aggregation, or trend comparison.
• Otherwise choose d = 0.

============================================================
CORE DECISION AGENT PROMPT — TEXT EVIDENCE ONLY
============================================================

Implement:

build_core_decision_text_prompt(
    question: str,
    instruction_set: str,
    evidence_summary: str,
    num_pages: int,
    instruction_md_text: str | None = None
) -> str

Use this prompt content:

You are an extractive QA model that gives answer to given query. You are given a query and a set of evidence. You have to provide specific answer from the given evidence, give your answer based only on the evidence. If you don’t find the answer within the evidence provided say 'No answers found!'. Use bullet points if you have to make a list, only if necessary. For counting questions, count carefully across all evidence. Mention which page the information came from.

QUERY:
{question}

STRATEGIC INSTRUCTIONS Γd:
{instruction_set}

EVIDENCE (from {num_pages} pages):
{evidence_summary}

YOUR ANSWER:

For implementation, require strict JSON output:

{
  "answer": "The shortest supported answer, or No answers found!",
  "evidence_references": [
    {
      "page_index": 0,
      "evidence": "Specific supporting evidence from the evidence context."
    }
  ]
}

Important:
• Answer based only on the provided evidence.
• If the evidence does not support an answer, output exactly:
  "No answers found!"
• Do not use outside knowledge.
• Do not infer beyond the evidence.
• For counting questions, count carefully across all evidence.
• For calculation questions, calculate only from provided values.
• Mention page indices in evidence_references.
• Return valid JSON only.
• Do not wrap the JSON in markdown.
• Do not output extra text.

============================================================
CORE DECISION AGENT PROMPT — WITH VISUALS
============================================================

Implement:

build_core_decision_visual_prompt(
    question: str,
    instruction_set: str,
    evidence_summary: str,
    visual_evidence_section: str,
    num_pages: int,
    instruction_md_text: str | None = None
) -> str

Use this prompt content:

You are an extractive QA model that gives answer to given query. You are given a query and evidence from relevant pages. You have to provide a specific, concise answer from the given evidence.

Instructions:
• Give your answer based only on the evidence provided.
• If you don’t find the answer within the evidence provided say 'No answers found!'.
• Provide ONLY the shortest possible answer: a number, a name, a short phrase, or a brief list - just the key information.
• Synthesize information across ALL pages of evidence when necessary (e.g., if one page has percentage A and another has percentage B, you may need to combine them).
• For calculation questions, perform the required calculations using data from the evidence.
• For counting questions, count carefully across all evidence.
• Use bullet points only if the answer is a list.

QUERY:
{question}

STRATEGIC INSTRUCTIONS Γd:
{instruction_set}

EVIDENCE (from {num_pages} pages):
{evidence_summary}

VISUAL EVIDENCE SECTION:
{visual_evidence_section}

YOUR ANSWER:

For implementation, require strict JSON output:

{
  "answer": "The shortest supported answer, or No answers found!",
  "evidence_references": [
    {
      "page_index": 0,
      "evidence": "Specific supporting evidence from the text or visual evidence."
    }
  ]
}

Important:
• Answer based only on provided evidence and provided page images.
• If the evidence does not support an answer, output exactly:
  "No answers found!"
• Do not use outside knowledge.
• Do not hallucinate page content.
• For counting questions, count carefully across all evidence and visible page content.
• For calculation questions, calculate only from provided values.
• Mention page indices in evidence_references.
• Return valid JSON only.
• Do not wrap the JSON in markdown.
• Do not output extra text.

============================================================
CORE DECISION PROMPT SELECTION
============================================================

Implement:

build_core_decision_prompt(
    question: str,
    instruction_set: str,
    evidence_summary: str,
    retained_page_indices: list[int],
    has_visuals: bool,
    instruction_md_text: str | None = None
) -> str

Behavior:
- If has_visuals is false or retained_page_indices is empty, call build_core_decision_text_prompt.
- If has_visuals is true, call build_core_decision_visual_prompt.

visual_evidence_section should look like:

The following page images are provided as visual evidence:
- Page index 3
- Page index 7

The actual images will be passed to the multimodal model separately.

============================================================
PROMPT TRACEABILITY REQUIREMENT
============================================================

In sleuth/agents/prompts.py, include a module-level docstring saying:

These prompts are adapted from Appendix F of:
"Resolving Evidence Sparsity: Agentic Context Engineering for Long-Document Understanding."

The original SLEUTH prompts used:
- Clue Discovery Agent
- Page Screening Agent
- Difficulty Assessment Agent
- Core Decision Agent (Text Evidence Only)
- Core Decision Agent (With Visuals)

This implementation keeps the same functional roles and prompt content, but enforces strict JSON outputs for reproducibility and parsing.

============================================================
AGENT PARSING REQUIREMENT
============================================================

Because these prompts request strict JSON, all agents must:
- use extract_json_from_text()
- validate with Pydantic schemas
- save raw_output always
- fall back safely when parsing fails

Fallbacks:
- Clue Discovery parse failure:
  has_relevant_evidence = false
  evidence_items = []
  page_summary = "Failed to parse clue discovery output."
  key_insights = ""

- Page Screening parse failure:
  has_visual_element = false
  relevance = "Irrelevant"
  reasoning = "Failed to parse page screening output."
  keep_page = false

- Difficulty parse failure:
  difficulty_level = 0
  instruction_set = "Use ordinary direct extraction from the provided evidence."

- Core Decision parse failure:
  answer = "No answers found!"
  evidence_references = []