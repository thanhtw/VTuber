# Research workflow for the theory-integrated manuscript

This implements the primary observational workflow in `Source/VTuber_Study1_CHB_Manuscript_v14_Theory_Integrated.pdf`, Sections 3.3–3.7 and Appendices A–D. The document's Section 4 values are synthetic templates: the program never uses them as findings. New empirical outputs live in `research_output/`, separate from the older `output/` exploratory files.

The workflow requires researcher decisions. It does not infer educational eligibility, fabricate inter-rater agreement, approve translation quality, or treat dictionary matches as validated psychological constructs.

## Install in Anaconda

```bash
conda activate VTuber
cd /home/selab/Desktop/Thomas/VTuber
# CPU installation; use your appropriate PyTorch build if you need a GPU.
python -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[research,dev]'
python -m nltk.downloader wordnet omw-1.4 vader_lexicon
```

In `.env` set `YOUTUBE_API_KEY` and `OPENAI_API_KEY`. Metadata and comments are translated with the configured `TRANSLATION_MODEL`. YouTube credentials are needed for discovery/collection; OpenAI credentials for translation. The pretrained English sentiment model downloads separately on first analysis. GPT translation incurs API charges.

```bash
python research_main.py check
# All commands also work as: python main.py research <command>
```

`research/study.json` controls retrieval and research analysis. `--study path/to/study.json` selects an alternative protocol, and `--env path/to/.env` selects credentials. Research output paths are relative to the study configuration file. Research settings do not replace the existing exploratory `.env` keyword settings.

## 1. Register retrieval and discover candidates

Edit `study.json` before collection. Each video query separately combines presenter identity with instruction, without restricting subjects. Separate queries avoid treating a teaching term as a Boolean OR alternative to VTuber identity. Channel discovery also searches explicit educational/teacher/knowledge VTuber descriptions. A query match is only a candidate, never automatic eligibility. Add translated query variants for relevant languages. `year_windows` can be an array such as `[2020, 2021, 2022, 2023, 2024, 2025, 2026]`; an empty array does not apply publication-date windows. Each configured query is expanded across its regions and ordering modes. Channel queries use `type=channel`.

```bash
python research_main.py discover
```

Discovery writes `candidates.json`, `channel_candidates.json`, and the two screening CSVs. Existing screening decisions are preserved on later discovery runs. Each API call is recorded in append-only `query_registry.jsonl` with run/call IDs, query context, parameters, UTC timestamp, pagination and returned IDs. Redacted response snapshots are retained in `api_responses/`. API keys and author profiles are excluded from snapshots/registry.

The default `max_search_pages_per_query=2` is a bounded pilot search; it is not exhaustive. Set it to `0` for pagination until the endpoint supplies no next page. YouTube search itself may omit eligible items; exhausted API pagination never proves complete population coverage. Run discovery on the separate collection dates specified in your protocol; the code logs actual dates and never invents a repeat search. Schedule runs according to your project's actual quota. Do not report a hard-coded quota claim from the manuscript without verifying it.

## 2. Independently verify channels and screen videos

Open `research_output/channel_screening.csv` and `screening.csv`. Both use:

| Field | Meaning |
|---|---|
| `id`, `title`, `channel_id` | Candidate identifiers; keep identifiers intact |
| `coder1`, `decision1` | First independent researcher and `include` / `exclude` |
| `coder2`, `decision2` | Different researcher and independent decision |
| `consensus` | Explicit `include` / `exclude` if decisions disagree |
| `exclusion_reason` | Required for excluded records |
| `format` | For included videos: `lecture`, `worked_example`, `livestream_lesson`, or `short` |
| `avatar_evidence` | Observation of the identifiable avatar acting as presenter |
| `instruction_evidence` | What knowledge/skill is taught and how it is explained, demonstrated or practiced |
| `instruction_timestamps` | Video time ranges supporting the instructional judgment |
| `notes` | Screening rationale or provenance |

The target population is **VTuber creators who teach knowledge or skills through an avatar presenter**. A mixed entertainment/education channel can qualify through its eligible lessons; every upload is still screened separately. Do not infer learner status or successful learning merely from audience comments.

Verify an identifiable avatar presenter and explicit instruction, demonstration, explanation or guided practice. Before including a video, record `avatar_evidence`, `instruction_evidence`, and `instruction_timestamps`; collection refuses included videos lacking these records. For example, “Avatar instructor demonstrates how to solve an equation; 02:15–06:40” describes evidence, whereas “the title says educational” does not. These fields are researcher observations, not automatic AI verification. Apply the manuscript's exclusions: non-avatar presenters, background-only avatars, fan edits/clips, advertisements, non-instructional gameplay, and tutorials about VTuber software. Channel identity and individual-video eligibility are separate decisions.

After channel verification, rerun `discover`. Upload playlists of included channels are paginated and their videos are added for screening. Researchers can also add discovered channel IDs to `channel_screening.csv` for documented snowball sampling; records still need both independent decisions. The code does not automatically crawl collaboration relationships or decide eligibility from video titles.

```bash
python research_main.py discover
python research_main.py screening-report
```

`screening-report` also adds new screening columns to existing CSVs without making API requests or changing recorded decisions. `screening_summary.json` reports pending/included/excluded counts, exclusion reasons, and Cohen's kappa from the two independent decisions before reconciliation. Kappa is missing when it cannot be estimated; blank forms do not count as agreement. Only videos with resolved inclusion, a valid manually assigned format, and a verified channel can be collected.

## 3. Collect screened metadata, top-level comments and replies

```bash
python research_main.py collect
```

The research collector fetches channels, full video metadata, every available top-level comment page, and all reply pages using `comments.list(parentId=...)`. Unlike exploratory collection, it does not impose the old 100-comment cap. It stores parent/video/channel relationships, creator-authorship flags, timestamps and like counts; commenter profiles are discarded.

Disabled/inaccessible comments and changed reply counts are recorded as coverage issues. Failures caused by credentials, quota or network errors stop collection. Per-video checkpoints and the API registry survive interruption. Rerun `collect` to finish: completed comment snapshots are reused, and unfinished videos are fetched again. An incomplete aggregate dataset cannot be analyzed.

Use a new study output directory for a new collection wave; cached completed checkpoints represent their original observation times. `collected_at` and `comments_collected_at` distinguish observations. The data are a time-specific public snapshot, not a guarantee that deleted/private material was captured.

## 4. Analyze RQ1, RQ2 and RQ3

```bash
python research_main.py analyze
```

Each completed analysis writes to a new timestamped `research_output/analyses/<run>/` directory. `latest_analysis.json` identifies the newest completed run. An interrupted analysis does not overwrite the last completed run.

### Translation and preprocessing

Both video textual metadata and comments are translated into English before their respective analyses. Original comment `text`, translated `text_en`, detected language, and declared video language are retained. One document per video combines title, description, tags and chapter titles parsed from timestamped description lines. The API does not supply a separate universal chapter-title field.

Configured boilerplate and URLs are removed before metadata translation; repeated description lines appearing in at least 80% of a channel's three or more videos are also removed. NLTK WordNet lemmatization, custom term mappings, English stop words, rare/frequent term filtering and frequent gensim bigrams prepare the LDA corpus. Review filters for small corpora rather than silently lowering them.

Stable coded channel/video/comment IDs are used in analysis and graph outputs. The restricted local `private_identifier_map.json` and `private_id_salt.json` support traceability. Text and video titles may remain identifiable even with coded IDs; these are restricted research artifacts, not public anonymized datasets. Output directory permissions are restricted and the directory is git-ignored. Do not publish raw comments or identifier maps as paper supplements.

### RQ1: instructional topics

Gensim online variational LDA fits the configured K and random-seed grid with learned asymmetric alpha and an eta/beta grid. A fixed held-out document subset supplies perplexity bounds for tuning eta within K. K ranking uses coherence (c_v), top-word diversity and seed stability using optimally matched topic-word Jaccard overlap. Perplexity is reported separately rather than used to rank K.

Without `selected_topic_count`, selection is explicitly provisional. Inspect diagnostics and representative videos, then set a defensible K, `selected_seed`, and researcher-approved labels/families in `study.json`:

```json
{
  "selected_topic_count": 4,
  "topic_names": {"1": "Researcher-reviewed label"},
  "topic_families": {"1": "Researcher-reviewed family"}
}
```

The labels above illustrate configuration, not discovered topics. Two researchers should label independently and resolve disagreements. The program also fits an original-English-only sensitivity model when enough English metadata are available. Videos whose filtered document is empty receive no topic assignment and contribute no topic weight, instead of an artificial uniform topic vector.

### RQ2: overall sentiment and nine aspects

The default is `cardiffnlp/twitter-roberta-base-sentiment-latest`, a documented pretrained English social-media model. Its resolved model commit and package versions are recorded. No classifier is trained on study outcomes. Long comments are scored in model-token windows and probabilities are averaged by window token count; no suffix is silently discarded. `sentiment_backend=vader` is permitted only as an explicit exploratory/sensitivity deviation and is marked in the manifest.

`aspect_rules.json` is a versioned pilot dictionary for:

| IDs | Observable evaluation aspect | Theoretical lens |
|---|---|---|
| A1–A3 | Organization, explanation, facilitation | CoI-informed instructional evaluation |
| A4 | Reported understanding or unresolved understanding | CoI-informed cognitive discourse |
| A5 | Peer interaction | CoI-informed social discourse |
| A6–A8 | Pacing, visual presentation, voice/delivery | CTML-informed multimedia evaluation |
| A9 | Persona/avatar, with appearance, companionship, attachment subtypes | PSI-informed persona discourse |

Multiple aspects can occur in one comment. All usable comments remain in overall estimates, including comments with no aspect. Rule matches and clause evidence are retained. Explicit local positive/negative rules assign aspect polarity; ambiguous mentions stay uninterpretable, and conflicting signs are mixed. Aspect polarity is never copied automatically from the comment's overall sentiment.

For topic k, each comment receives its video's topic proportion theta_vk. Overall polarity share is sum(theta_vk * polarity_indicator) / sum(theta_vk). Aspect prevalence uses all usable comments in the denominator. Aspect polarity conditional on mention uses only the corresponding aspect mentions; uninterpretable and mixed are explicit categories. Zero denominators are missing, not zero percentages. Intervals resample entire channels; fewer than two contributing channels produce missing intervals.

### RQ3: associations and networks

The binary comment-by-aspect incidence matrix is the auditable starting point. For each of all 36 unordered pairs, export observed co-mentions O_ab, expected E_ab=N*p_a*p_b, O/E, cooccurrence proportion, channel bootstrap intervals and null comparisons. Weighted topic/family contexts replace counts with sums of topic weights and retain weighted denominators. Same-video mentions in separate comments never create a same-comment pair.

The main null permutes each aspect column within channels, preserving channel-level aspect margins. A separate checkerboard-swap sensitivity preserves comment degrees and within-channel column margins; finite swaps are an approximate randomization, and movement counts are reported. Degenerate null distributions have no inferential p-value. Benjamini–Hochberg adjusted q values are supplied within networks and across reported networks. Review sparse cells and cluster counts before interpretation.

Contexts include formats, original comment languages, soft-weighted topics and manually assigned topic families. Sensitivity outputs exclude the largest channel, use strict dictionary rules, separate original-English/non-English comments, and give each video's comments equal total weight. Direct context contrasts use shared channel bootstrap draws to estimate ratio differences; these exploratory contrast intervals are unadjusted and should not be reported as multiplicity-corrected significance tests.

The heterogeneous GraphML connects channel → video → comment → aspects, plus topic-probability, format, language and reply links. Aspect network edge weight is positive log2(O/E), distinct from raw frequency; depleted pairs remain in the CSV. Greedy modularity communities are exploratory, compared with fixed-margin null modularity and channel-bootstrap coassignment stability. There is no GNN: no independent future prediction target has been specified.

## Outputs mapped to manuscript results

| Manuscript result | Computed artifact under the analysis run |
|---|---|
| Sample overview / Table 5 | `table5_sample_overview.csv`, `sample_overview.json`, root `screening_summary.json` |
| Topic families / Table 6 | `table6_topic_summary.csv`, `rq1/topics.json`, `rq1/representative_videos.csv` |
| LDA Appendix B | `rq1/model_grid.csv`, `model_selection.csv`, `selection.json`, saved model/dictionary |
| Overall sentiment / Table 7 | `overall_sentiment_by_context.csv` (weighted numerator, denominator, estimate, CI) |
| Aspect results / Table 8 | `aspect_prevalence_and_polarity.csv`, `a9_subtypes.csv`, precision-audit results |
| Pair associations / Table 9 | `rq3/aspect_associations.csv`, `comment_aspect_incidence.csv`, `pair_evidence.csv` |
| Robustness / Table 10 | Sensitivity contexts in associations CSV; `rq3/context_contrasts.csv` |
| Relational configurations | `rq3/heterogeneous.graphml`, `aspect_network.graphml`, `aspect_network.png`, `communities.json`, `community_stability.csv` |
| Optional Table 11 | Omitted; the manifest records why no neural prediction model was trained |
| Original and translated texts | `comments.csv`, `analyzed_dataset.json` |
| Descriptive metadata | `channels.csv`, `video_metadata.csv`, `video_response_indicators.csv` |
| Reproducibility | `manifest.json`: data/settings/code hashes, package versions, models, rule hash and deviations |

## Independent audits and subsequent reruns

The analysis creates `translation_review_sample.csv` stratified by original language and `aspect_precision_audit_sample.csv` stratified by predicted aspect. These are blank review sheets, not validation evidence. Ask independent bilingual researchers to complete the translation review. To apply agreed corrections, copy the completed sample to `research_output/translation_reviews.csv`: distinct `reviewer1` and `reviewer2`, `reviewed_text_en`, coded `comment_id`, and the unchanged `original_sha256` are required. Corrected text is applied before scoring and aspect extraction on the next run.

Copy the completed aspect audit to `research_output/aspect_precision_audit.csv`. Record distinct reviewer IDs and `correct1`/`correct2` as `0` or `1`; resolve disagreements with `consensus_correct`. Reruns summarize precision with Wilson intervals, explicitly without claiming recall. The unchanged `rules_sha256` and `text_en_sha256` must match the current dictionary and translated text, and audited IDs must still be predicted positives. Stale rule versions are excluded; re-review after rule changes. Topic labeling, human-format assignment and ethical review remain research responsibilities.

## Method limits that must be disclosed

- No collapsed Gibbs robustness sampler is implemented, despite Appendix B's proposed check; remove that claim or implement a separately validated check before submission.
- Human screening, translation quality and aspect precision cannot be established by running software. Until completed, findings remain provisional.
- Appearance, companionship and attachment language are observable expressions, not latent parasocial-bond measurements.
- Sentiment measures public discourse, not actual learning, satisfaction or causal mechanisms. No conventional-presenter comparison is implemented.
- Broad discovery, full channel-upload screening, repeated-date coverage and multilingual query coverage depend on the protocol you actually execute. The default bounded English pilot cannot substantiate a representative multilingual census.
- Subscriber/like counts can be missing; missing values are preserved. Share of lessons among **all** uploads remains unestimated until every relevant upload has been screened; included-video count is not a substitute.
- Contextual networks and the finite-swap community procedure are exploratory. Report channel replication, sparse cells, multiplicity and uncertainty; do not turn a visually strong edge into a causal claim.

References for implementation: [YouTube replies](https://developers.google.com/youtube/v3/docs/comments/list), [gensim LDA](https://radimrehurek.com/gensim/models/ldamodel.html), [pretrained model card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), [PyTorch installation](https://pytorch.org/get-started/locally/).

## Scope and correctness for identifying teaching VTubers

Use `python research_main.py discover`, not the exploratory `python main.py` search, for this study. The research configuration and channel/video evidence checks determine the research corpus; the old generic `SEARCH_KEYWORD=VTuber` corpus is not automatically a corpus of educational lessons.

The current English/US/GB configuration with two search pages is a pilot. It cannot establish worldwide coverage, and creators may teach without labeling themselves “educational VTuber”. Expand multilingual queries, document channel snowballing, inspect verified-channel uploads, and repeat searches on the registered dates. Avoid restricting queries to specific subjects because RQ1 is meant to discover subjects. Educational eligibility requires actual content review, not LDA topic labels, sentiment positivity, or keyword hits.

## AI-assisted first-pass review of the screening files

`review_screening.py` reviews previously retrieved metadata in `research_output/ai_screening_review/videos_metadata.json` and `channels_metadata.json`. It uses the configured GPT model, caches completed batches, and adds `ai_*` columns to both screening CSVs while preserving all human decisions. It does not obtain missing metadata automatically; metadata must be retrieved first. Metadata reviewed by the assistant are retained with their API provenance.

- `ai_recommendation=advance`: promising candidate for actual content review, not confirmed eligibility.
- `ai_recommendation=exclude`: a proposed exclusion supported by metadata; a researcher should verify it.
- `ai_recommendation=uncertain`: insufficient or conflicting evidence; do not interpret it as exclusion.
- `ai_metadata_evidence`, `ai_avatar_metadata_quote`, and `ai_instruction_metadata_quote` contain quotations checked against supplied metadata.
- `ai_suggested_format` is provisional and does not replace the reviewed `format` field.
- `ai_additional_evidence_check` distinguishes thumbnail inspection from actual video viewing.

Backups, review queues, model caches and a count summary are saved under `ai_screening_review/`. No `coder1`, `coder2`, `decision1`, `decision2`, `consensus`, verified teaching evidence or viewing timestamps are filled by this AI metadata pass. Never describe this as two independent human reviews or use it to manufacture screening kappa. Complete actual-content verification before admitting videos to the manuscript dataset.
