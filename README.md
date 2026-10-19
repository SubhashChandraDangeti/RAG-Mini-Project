# RAG Assignment: Baseline vs. Cross-Encoder Re-ranking

The implementation is in `rag_baseline_vs_reranking.ipynb`. It uses `US_Chess_RuleBook.pdf` as the single document corpus and follows the assignment's fixed Top 10 / Top 3 design.

## Project files and results

- [Executed notebook](rag_baseline_vs_reranking.ipynb): code, retrieved passages, generated answer comparisons, timings, and validation checks.
- [Report draft](REPORT.md): measured results and prompts; the interpretation paragraphs and student details still need to be completed by the author.
- [Git command guide](GIT_GUIDE.md): commands used to upload the project and update it later.
- [Dependencies](requirements.txt): packages needed to run the notebook.

The saved experiment uses `gemma-4-26b-a4b-it` through the Google GenAI client. Q3 promotes its reference chunk from rank 10 to 1; Q6 promotes its reference chunk from rank 4 to 1.

## Download the corpus

Download the [US Chess official free-download rulebook, revised 21 August 2020](https://new.uschess.org/sites/default/files/media/documents/us-chess-rule-book-online-only-edition-chapters-1-2-10-11-9-1-20.pdf).

Save it in the project folder as `US_Chess_RuleBook.pdf`. This is the 139-page free-download edition containing Chapters 1, 2, 10, and 11. The PDF remains copyright US Chess and is not redistributed in this repository. The experiment answers questions about this document edition.

## Run the notebook

1. Open a terminal in this folder.
2. Optionally create and activate a virtual environment.
3. Install the dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Set a Gemini API key for answer generation:

   ```powershell
   $env:GEMINI_API_KEY = "your-key"
   ```

5. Start Jupyter and run every notebook cell in order:

   ```powershell
   python -m notebook
   ```

The embedding and Cross-Encoder models are downloaded from Hugging Face on their first use. Later runs use the local model cache.

## What the notebook verifies

- The baseline uses a dense Bi-Encoder and cosine similarity.
- It retrieves exactly 10 chunks and sends its first 3 to the LLM.
- The Cross-Encoder scores exactly those same 10 candidates.
- Re-ranking selects a new Top 3 and uses the identical LLM prompt.
- Chess-specific stress tests cover exceptions, negation, distractors, and constraints.
- Rank shifts and latency are presented as tables.
- A final validation cell requires at least two genuine rank 4-10 to Top 3 promotions.

Retrieval rankings can vary slightly with model and library versions. If the validation cell finds fewer than two qualifying cases, inspect the provided rank tables and refine the closest stress-query wording before submission.

Complete the bracketed prompts in `REPORT.md` in your own words before final submission. The report is currently a draft.
