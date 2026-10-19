# RAG Assignment Report: Results Worksheet

> Measured facts and evidence from the completed run are filled in below. Write the interpretation paragraphs yourself, then remove the prompts and this note. The assignment explicitly requires a report in your own words.

Experiment source: [completed notebook](rag_baseline_vs_reranking.ipynb).

Name: [your name] | Roll number: [your roll number] | GitHub repository: https://github.com/SubhashChandraDangeti/RAG-Mini-Project

## Experimental setup

| Item | Recorded configuration/result |
| --- | --- |
| Document | US Chess Federation's Official Rules of Chess, 7th Edition; free-download version revised 21 August 2020 |
| Document scope | Chapters 1, 2, 10, and 11; 139 PDF pages |
| Content retained | 126 pages; 405,752 extracted characters |
| Preprocessing | Start at PDF page 13; remove repeated headers/footers, isolated page numbers, and pages with fewer than 150 cleaned characters |
| Chunking | Page-local windows of 190 words, with 40-word overlap |
| Indexed chunks | 484, with stable chunk IDs and PDF-page metadata |
| Bi-Encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector index | In-memory NumPy matrix of normalized dense embeddings; exact cosine-similarity search |
| Embedding dimensions | 484 x 384 |
| Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Answer-generation model | `gemma-4-26b-a4b-it`, accessed through the Google GenAI client |
| Temperature | 0.0 |
| Baseline context | First 3 chunks from the dense Top 10 |
| Re-ranked context | First 3 after scoring the same 10 query-chunk pairs with the Cross-Encoder |
| Experimental controls | Same PDF, chunks, initial candidate set, LLM, temperature, prompt template, and final context count |
| Hardware | [enter the CPU/GPU used for this run] |

The evaluation uses this specific 2020 document as its source; it does not establish current tournament rules.

## 1. Architecture comparison

[Write a short paragraph explaining how query and document embeddings are produced independently by the Bi-Encoder, why normalized dot product equals cosine similarity, and how the Cross-Encoder jointly processes query and passage tokens through self-attention. Explain the accuracy/speed trade-off in your own words.]

## 2. Qualitative and rank-shift analysis

### Recorded evidence promotions

| Query | Failure category | Evidence chunk | PDF page | Baseline rank | Re-ranked position |
| --- | --- | --- | ---: | ---: | ---: |
| Q3 | Corner case / event ordering | `P040-C01` | 40 | 10 | 1 |
| Q6 | Exception / equipment | `P022-C01` | 22 | 4 | 1 |

Both evidence chunks were already among the original ten candidates but outside the baseline Top 3.

### Q3: Does checkmate count when both players have no time left?

| Pipeline | Final context, in order |
| --- | --- |
| Baseline | `P031-C01`, `P034-C02`, `P026-C01` |
| Re-ranked | `P040-C01`, `P034-C03`, `P031-C01` |

Evidence to discuss:

- Baseline Answer 1 cites the general priority of checkmate over a subsequent flag fall. It does not mention the both-flags-down draw claim before checkmate.
- Re-ranked Answer 2 includes the sudden-death condition from `P040-C01`: a claim that both flags are down before checkmate makes the game a draw.
- Rule 14G1 distinguishes the timing of the **both-flags-down claim relative to checkmate**, rather than simply the physical flag fall. Use that precise distinction when evaluating the generated answers.

[Write your own explanation of why the general-rule passages did not fully cover this corner case, and how the rank 10 -> 1 shift changed the supplied evidence. Classify the baseline issue accurately: omission of a critical condition, rather than an invented rule. Check both answers against PDF page 40 before judging correctness.]

### Q6: After a chess game has started, can its clock ever be replaced, and how must the replacement be configured?

| Pipeline | Final context, in order |
| --- | --- |
| Baseline | `P129-C02`, `P121-C03`, `P029-C03` |
| Re-ranked | `P022-C01`, `P121-C03`, `P049-C04` |

Evidence to discuss:

- Baseline Answer 1: "The provided context is insufficient."
- Re-ranked Answer 2 identifies defective-clock replacement and references the exceptions in rules 5F5 and 14H2a.
- It also states that a replacement for a defective clock must be set in the same manner as the clock being replaced, supported by `P022-C01`.

[Write your own explanation connecting the rank 4 -> 1 promotion to the improved answer. Note that the baseline abstained; it did not hallucinate.]

### Other stress tests and limitations

| Test | Baseline reference rank | Re-ranked reference rank | Required promotion demonstrated? |
| --- | ---: | ---: | --- |
| Q1: Rook touched first | 1 | 1 | No; evidence was already in Top 3 |
| Q2: Director intervention in time pressure | Outside Top 10 | Outside Top 10 | No; re-ranking cannot retrieve a missing candidate |
| Q3: Checkmate and expired clocks | 10 | 1 | Yes |
| Q4: Unavoidably delayed arrival | 1 | 1 | No; evidence was already in Top 3 |
| Q5: Physical clocks in online play | 1 | 1 | No; evidence was already in Top 3 |
| Q6: Clock replacement | 4 | 1 | Yes |
| Q7: Promotion-piece commitment | 1 | 3 | No; reference evidence moved down but remained in Top 3 |
| Q8: Withdrawing a draw offer | 1 | 1 | No; evidence was already in Top 3 |

[If discussing these tests, distinguish the designated reference chunk from all potentially relevant chunks. The evidence check uses specific phrase pairs, not a complete human relevance annotation. The query bank was deliberately stress-tested and Q3 was refined; these are illustrative failure cases, not an unbiased estimate of performance on all chess questions.]

## 3. Latency profiling

Timing protocol: one warm-up per query, then five repetitions. Each component is reported as its median. The total is the sum of the baseline and re-ranking medians. Model loading, initial document embedding, and LLM generation are excluded.

| Query | Baseline retrieval (ms) | Cross-Encoder addition (ms) | Retrieval + re-ranking (ms) |
| --- | ---: | ---: | ---: |
| Q1 | 22.20 | 429.04 | 451.24 |
| Q2 | 17.63 | 432.63 | 450.26 |
| Q3 | 15.11 | 358.72 | 373.84 |
| Q4 | 20.37 | 389.59 | 409.96 |
| Q5 | 16.48 | 382.99 | 399.46 |
| Q6 | 16.78 | 382.18 | 398.97 |
| Q7 | 17.34 | 373.43 | 390.76 |
| Q8 | 21.11 | 383.95 | 405.06 |
| Mean across query medians | 18.38 | 391.57 | 409.94 |

Values are rounded independently to two decimals; displayed components can differ from their displayed total by 0.01 ms.

[Interpret the extra latency alongside the two observed relevance improvements. Do not present these timings as end-to-end answer-generation latency; temporary API overloads and generation retries are outside this measurement.]

## 4. Scaling question

[Explain in your own words why we do not score the full collection with a Cross-Encoder first. Address document-vector precomputation, query-dependent joint inference, the number of query-chunk pairs, and how fast dense retrieval reduces the expensive scoring stage to ten pairs. You may compare scoring 484 pairs for this corpus with scoring only ten.]

## Submission details

- Include the executed notebook and required source/dependency files in the public or shared GitHub repository.
- Verify that the notebook contains both Q3/Q6 answer pairs, the timing results, and the passed Section 11 checks.
- Replace the bracketed prompts with your own concise prose and include your repository URL.
