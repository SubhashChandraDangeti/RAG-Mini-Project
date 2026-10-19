# RAG: Baseline Retrieval vs. Cross-Encoder Re-ranking

**Name:** D. Subhash

**Roll number:** 2025201007

**Repository:** [RAG-Mini-Project](https://github.com/SubhashChandraDangeti/RAG-Mini-Project)

**Implementation and saved results:** [Jupyter notebook](rag_baseline_vs_reranking.ipynb)



## Experimental objective and setup

The objective was to determine whether a Cross-Encoder could improve the relevance of the passages supplied to an LLM after initial dense retrieval. Two RAG pipelines were compared over the same document. The baseline supplied the first three dense-retrieval results to the LLM, while the advanced pipeline re-ranked the same ten candidates before selecting three.

The corpus was the *US Chess Federation's Official Rules of Chess, 7th Edition*, free-download version revised on 21 August 2020. It contains Chapters 1, 2, 10, and 11 across 139 PDF pages. Its general rules, exceptions, tournament variations, and separate online-play provisions provide closely related passages that can be difficult to distinguish through semantic similarity alone. The experiment concerns this specific document edition.

| Component | Configuration |
| --- | --- |
| PDF extraction | PyMuPDF, with text sorted by page position |
| Content retained | 126 pages and 405,752 cleaned characters |
| Chunking | Up to 190 words per chunk, with 40-word overlap, within each page |
| Corpus size | 484 chunks |
| Chunk metadata | Stable chunk ID and original PDF page number |
| Bi-Encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Dense index | In-memory NumPy matrix; 484 vectors of 384 dimensions |
| Similarity | Dot product of L2-normalized vectors, equivalent to cosine similarity |
| Initial retrieval | Top 10 candidates |
| Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Final context | Top 3 chunks in both pipelines |
| Answer model | `gemma-4-26b-a4b-it`, through the Google GenAI client |
| Generation temperature | 0.0 |

Extraction started at PDF page 13 to exclude the cover, update summaries, and table of contents. Cleaning removed repeated headers and footers, isolated page numbers, soft hyphens, and line-break hyphenation. Pages containing fewer than 150 cleaned characters were excluded. Chunks retained their page numbers so retrieved evidence could be traced back to the PDF.

Both pipelines used the same corpus, chunk boundaries, initial ten candidates, LLM, prompt template, temperature, and final context count. The prompt requested a concise answer supported by chunk/page citations and instructed the model to acknowledge insufficient context. The experimental difference was the selection and ordering of the final three chunks.

Corpus source: [US Chess rulebook PDF included in the repository](US_Chess_RuleBook.pdf). The notebook reads this file from the project folder.

## 1. Architecture comparison

### Bi-Encoder: fast candidate retrieval

The Bi-Encoder processes a query and a document chunk independently, converting each into a fixed-size vector. Document vectors are computed once during indexing. At query time, the system embeds the question and compares its vector against the stored chunk vectors.

Cosine similarity measures the alignment of two vectors:

```text
cosine(q, d) = (q · d) / (||q|| × ||d||)
```

The implementation normalizes both query and chunk embeddings to unit length, so their dot product directly gives cosine similarity. It sorts these scores and retrieves the ten highest-scoring chunks.

This design is efficient because document embeddings can be reused for every question. However, semantic similarity does not necessarily indicate that a passage answers a specific question. Passages about the same subject may differ in a crucial exception, timing condition, or role restriction.

### Cross-Encoder: detailed relevance scoring

The Cross-Encoder receives each query and candidate chunk together. Joint self-attention allows tokens in the question to interact with tokens in the passage before the model produces a relevance score. This supports finer distinctions than comparing independently compressed vectors.

The implementation scores all ten retrieved query–chunk pairs, sorts them in descending score order, and supplies the new Top 3 to the LLM. Cross-Encoder scores and cosine scores have different scales; their numerical magnitudes are not compared directly.

| Aspect | Baseline pipeline | Re-ranking pipeline |
| --- | --- | --- |
| Candidate retrieval | Bi-Encoder Top 10 | Same Bi-Encoder Top 10 |
| Final selection | First 3 dense results | First 3 after joint relevance scoring |
| Additional model inference | None | Cross-Encoder over 10 pairs |
| Main advantage | Low retrieval latency | More precise context selection |
| Main limitation | May miss an important clause in final context | Additional latency; cannot recover absent candidates |

## 2. Qualitative and rank-shift analysis

Eight stress queries covered exceptions, negation, equipment restrictions, promotion, online play, and event ordering. Reference chunks were identified using document-specific evidence phrases. Two cases met the required condition: the critical evidence was initially at ranks 4–10 and moved into the re-ranked Top 3.

| Query | Failure mode | Critical evidence | Baseline rank | Re-ranked rank |
| --- | --- | --- | ---: | ---: |
| Q3 | Corner case and event ordering | `P040-C01`, PDF page 40 | 10 | 1 |
| Q6 | General rule versus equipment exceptions | `P022-C01`, PDF page 22 | 4 | 1 |

### Case 1: Checkmate when both players have no time left

**Query:** “Does checkmate count when both players have no time left?”

| Pipeline | Context chunks, in order |
| --- | --- |
| Baseline | `P031-C01`, `P034-C02`, `P026-C01` |
| Re-ranked | `P040-C01`, `P034-C03`, `P031-C01` |

**Baseline Answer 1.** The answer explained that checkmate takes priority over a subsequent flag fall and ends the game without requiring the player to stop the clock. It also discussed a time-forfeit claim made before checkmate. However, it omitted the specific sudden-death condition concerning a claim that *both* flags were down.

**Re-ranked Answer 2.** After `P040-C01` moved from rank 10 to rank 1, the answer included the missing condition: in sudden death, if a player points out that both flags are down before checkmate, the game is drawn. It also discussed director discretion and the distinction involving time controls with increments of 30 seconds or more.

**Interpretation.** The baseline retrieved broadly relevant checkmate and time-forfeit passages but excluded the passage resolving the specific corner case. This was an omission of an essential condition, rather than evidence of an invented rule. Re-ranking improved the context by promoting the rule that directly addressed the interaction between both flags being down and checkmate.

The generated answer still needs careful reading. Rule 14G1 concerns whether the **both-flags-down claim occurred before checkmate**. Answer 2 describes its unclear-timing condition using “flag fall,” which is less precise. Thus, retrieval improved, but the result does not establish that every sentence of the generated answer is perfectly faithful.

### Case 2: Replacing a clock after play has started

**Query:** “After a chess game has started, can its clock ever be replaced, and how must the replacement be configured?”

| Pipeline | Context chunks, in order |
| --- | --- |
| Baseline | `P129-C02`, `P121-C03`, `P029-C03` |
| Re-ranked | `P022-C01`, `P121-C03`, `P049-C04` |

**Baseline Answer 1.** The model responded: “The provided context is insufficient.” Although the retrieved passages were related to clocks or time controls, the final three chunks did not contain the required replacement rule.

**Re-ranked Answer 2.** The answer stated that a defective clock may be replaced, referenced the exceptions in rules 5F5 and 14H2a, and explained that a replacement for a defective clock must be set in the same manner as the original clock. These statements were supported by `P022-C01`.

**Interpretation.** The decisive passage was present at baseline rank 4, immediately outside the context cutoff. Promoting it to rank 1 changed the result from abstention to a supported answer. The improvement came from selecting existing evidence, without retrieving new candidates or changing the prompt. The baseline behaved appropriately by abstaining; its limitation was inadequate context, not hallucination.

### Results across the query bank

| Test | Topic | Baseline reference rank | Re-ranked reference rank |
| --- | --- | ---: | ---: |
| Q1 | Rook touched first when castling | 1 | 1 |
| Q2 | Director intervention during time pressure | Outside Top 10 | Outside Top 10 |
| Q3 | Checkmate and expired clocks | 10 | 1 |
| Q4 | Unavoidably delayed arrival | 1 | 1 |
| Q5 | Physical clocks in online play | 1 | 1 |
| Q6 | Clock replacement | 4 | 1 |
| Q7 | Commitment to a promotion piece | 1 | 3 |
| Q8 | Withdrawing a draw offer | 1 | 1 |

Re-ranking did not improve every reference rank. Several passages were already in the baseline Top 3, while Q7's reference passage moved down but remained included. Q2's designated reference chunk was missing from the initial Top 10, so the Cross-Encoder could not recover it.

The reference test uses phrase pairs rather than exhaustive human relevance labels. “Outside Top 10” therefore refers to the designated reference evidence, not proof that every retrieved passage was irrelevant. The queries were deliberately stress-tested, and Q3 was refined during exploration. These results illustrate failure modes; they are not an unbiased accuracy estimate for arbitrary chess questions.

## 3. Latency profiling

Each query received one warm-up run followed by five timed repetitions. Timing used `time.perf_counter()`. The baseline measurement included query embedding and dense search; the additional measurement covered Cross-Encoder scoring and sorting.

The table reports the median of each component. The two-stage total is the **sum of those component medians**. Model loading, document embedding/index construction, and LLM generation were excluded.

| Query | Baseline retrieval (ms) | Re-ranking addition (ms) | Two-stage total (ms) |
| --- | ---: | ---: | ---: |
| Q1 | 22.20 | 429.04 | 451.24 |
| Q2 | 17.63 | 432.63 | 450.26 |
| Q3 | 15.11 | 358.72 | 373.84 |
| Q4 | 20.37 | 389.59 | 409.96 |
| Q5 | 16.48 | 382.99 | 399.46 |
| Q6 | 16.78 | 382.18 | 398.97 |
| Q7 | 17.34 | 373.43 | 390.76 |
| Q8 | 21.11 | 383.95 | 405.06 |
| **Mean across query medians** | **18.38** | **391.57** | **409.94** |

Values are independently rounded, so a displayed total can differ from the sum of its displayed components by 0.01 ms.

Across the eight queries, re-ranking added approximately 392 ms. The mean two-stage retrieval time was approximately 410 ms, compared with 18 ms for baseline retrieval. The ratio of these rounded means is about 22.3, showing that joint relevance scoring dominated retrieval time in this experiment.

For Q3 and Q6, this extra computation brought the essential evidence into the final context. Whether that trade-off is worthwhile depends on the application: a rule-explanation system may benefit from including a critical exception even at the cost of several hundred additional milliseconds.

These are retrieval timings, not end-to-end answer times. Temporary API overloads and retries affected answer generation separately. The saved run did not record CPU/GPU specifications or device selection, so the timings should be treated as measurements of the local Windows/Jupyter environment rather than hardware-independent performance claims.

## 4. Why not apply a Cross-Encoder to the entire collection?

A Cross-Encoder's score depends on the particular query and passage together. Unlike a stored document embedding, that score cannot be precomputed once and reused for unrelated questions.

For a collection of `N` chunks, scoring the full collection requires `N` query–chunk evaluations for every new question. In this implementation, dense retrieval first reduces the collection to ten candidates, so only ten pairs require the more expensive joint inference.

| Retrieval design | Cross-Encoder pairs per query |
| --- | ---: |
| Score every chunk in this corpus | 484 |
| Re-rank the dense Top 10 | 10 |
| Score every chunk in a hypothetical million-chunk corpus | 1,000,000 |

For the present corpus, full Cross-Encoder scoring would involve 48.4 times as many pairs as re-ranking ten candidates. This is a comparison of pair counts, not a measured runtime multiplier; batching, passage lengths, and hardware influence actual latency.

The Bi-Encoder supports this separation by reusing document vectors. The current NumPy index performs exact search over all 484 vectors. Larger systems can use approximate nearest-neighbour indexes to reduce vector-search cost. The resulting architecture combines efficient candidate discovery with more detailed scoring of a small set.

Its central limitation remains candidate recall: if dense retrieval does not place the required evidence within the Top 10, re-ranking cannot introduce it. Q2 demonstrates that limitation for the designated reference chunk.

## Conclusion

The experiment demonstrated two concrete benefits of Cross-Encoder re-ranking. Q3's critical corner-case passage moved from rank 10 to rank 1, adding a missing draw condition. Q6's clock-replacement passage moved from rank 4 to rank 1, enabling an evidence-supported answer where the baseline had abstained.

The measured benefit came with additional retrieval latency of approximately 392 ms on average. Re-ranking improved context selection in the two selected cases, but did not improve every query and did not guarantee flawless generation. The results support using dense retrieval for candidate discovery and a Cross-Encoder for focused relevance refinement when important exceptions justify the extra computation.

