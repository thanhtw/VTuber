"""RQ1: lemmatized metadata, gensim LDA grid, coherence/diversity/stability, and held-out perplexity."""
import re
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from .storage import table, save


def chapter_titles(description):
    return [match.group(1).strip() for match in re.finditer(r"(?m)^\s*(?:\d{1,2}:)?\d{1,2}:\d{2}\s+(.+)$", description)]


def clean_metadata(videos, study):
    by_channel = defaultdict(list)
    for video in videos:
        by_channel[video['channel_id']].append(video)
    repeated = {}
    for channel, records in by_channel.items():
        counts = Counter(line.strip() for v in records for line in set(v['description'].splitlines()) if len(line.strip()) > 25)
        repeated[channel] = {line for line, count in counts.items() if len(records) >= 3 and count/len(records) >= .8}
    result = []
    for video in videos:
        description = '\n'.join(line for line in video['description'].splitlines() if line.strip() not in repeated[video['channel_id']])
        text = '\n'.join([video['title'], description, ' '.join(video.get('tags', [])), ' '.join(video.get('chapter_titles', []))])
        for pattern in study.get('boilerplate_patterns', []):
            text = re.sub(pattern, ' ', text)
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        result.append(text)
    return result


def topic_similarity(left, right):
    matrix = np.array([[len(a & b)/len(a | b) if a | b else 0 for b in right] for a in left])
    rows, columns = linear_sum_assignment(-matrix)
    return float(matrix[rows, columns].mean())


def fit_topics(videos, study, output):
    try:
        from gensim.corpora import Dictionary
        from gensim.models import LdaModel, CoherenceModel, Phrases
        from nltk.stem import WordNetLemmatizer
    except ImportError:
        raise ValueError('Install: python -m pip install -e ".[research]"') from None
    lemmatizer = WordNetLemmatizer()
    try:
        lemmatizer.lemmatize('lessons')
    except LookupError:
        raise ValueError('Run: python -m nltk.downloader wordnet omw-1.4') from None
    terms = study.get('term_dictionary', {})
    tokens = [[terms.get(word, lemmatizer.lemmatize(word)) for word in re.findall(r"[a-z][a-z_]+", v['metadata_text_en'].lower())
               if word not in ENGLISH_STOP_WORDS] for v in videos]
    phrases = Phrases(tokens, min_count=study.get('bigram_min_count', 3), threshold=10)
    tokens = [list(phrases[words]) for words in tokens]
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=study.get('min_document_frequency', 2), no_above=study.get('max_document_fraction', .8))
    corpus = [dictionary.doc2bow(words) for words in tokens]
    usable = [i for i, bow in enumerate(corpus) if bow]
    if len(usable) < 4 or len(dictionary) < 2:
        raise ValueError('RQ1 needs at least four videos with usable metadata and two retained terms. Collect more eligible videos or review frequency filters; no topics were fabricated.')
    rng = np.random.default_rng(study.get('seed', 42))
    shuffled = rng.permutation(usable)
    holdout_count = max(1, len(usable)//5)
    holdout = [corpus[i] for i in shuffled[:holdout_count]]
    training = [corpus[i] for i in shuffled[holdout_count:]]
    training_texts = [tokens[i] for i in shuffled[holdout_count:]]
    candidates = [k for k in study.get('topic_counts', [2, 3, 4, 5]) if k <= min(len(training), len(dictionary))]
    if not candidates:
        raise ValueError('No configured topic count fits the training corpus.')
    topn = min(study.get('topic_top_words', 10), len(dictionary))
    rows, runs = [], {}
    for k in candidates:
        for eta in study.get('eta_grid', [.01, .1]):
            for seed in study.get('seeds', [42, 123, 2026]):
                model = LdaModel(corpus=training, id2word=dictionary, num_topics=k, random_state=seed,
                                 alpha='auto', eta=eta, update_every=1,
                                 passes=study.get('lda_passes', 10), iterations=study.get('lda_iterations', 100),
                                 minimum_probability=0.)
                words = [set(word for word, _ in model.show_topic(i, topn)) for i in range(k)]
                coherence = float(CoherenceModel(model=model, texts=training_texts, dictionary=dictionary,
                                                coherence='c_v', topn=topn, processes=1).get_coherence())
                bound = float(model.log_perplexity(holdout))
                rows.append({'k': k, 'eta': eta, 'seed': seed, 'coherence_cv': coherence,
                             'diversity': len(set.union(*words))/(k*topn),
                             'heldout_log2_perplexity_bound': bound,
                             'heldout_perplexity_bound': float(np.exp2(-bound)),
                             'training_documents': len(training), 'heldout_documents': len(holdout)})
                runs[k, eta, seed] = words
    summaries = []
    for k in candidates:
        for eta in study.get('eta_grid', [.01, .1]):
            matched = [r for r in rows if r['k'] == k and r['eta'] == eta]
            similarities = [topic_similarity(runs[k, eta, a], runs[k, eta, b]) for a, b in combinations(study.get('seeds', [42, 123, 2026]), 2)]
            summaries.append({'k': k, 'eta': eta, 'mean_coherence_cv': float(np.nanmean([r['coherence_cv'] for r in matched])),
                              'mean_diversity': float(np.mean([r['diversity'] for r in matched])),
                              'mean_heldout_log2_bound': float(np.mean([r['heldout_log2_perplexity_bound'] for r in matched])),
                              'seed_stability_jaccard': float(np.mean(similarities)) if similarities else None})
    # Tune beta/eta on held-out diagnostics within each K. K itself is not selected by perplexity.
    eligible_summaries = []
    for k in candidates:
        best_eta = max([r for r in summaries if r['k'] == k], key=lambda r: r['mean_heldout_log2_bound'])['eta']
        for row in summaries:
            if row['k'] == k:
                row['eta_selected_on_holdout'] = row['eta'] == best_eta
                if row['eta_selected_on_holdout']:
                    eligible_summaries.append(row)
    # Provisional ranking uses all three diagnostics. Human confirmation is explicitly retained.
    for metric in ['mean_coherence_cv', 'mean_diversity', 'seed_stability_jaccard']:
        ordered = sorted(eligible_summaries, key=lambda r: r[metric] if r[metric] is not None and np.isfinite(r[metric]) else -np.inf, reverse=True)
        for rank, row in enumerate(ordered):
            row.setdefault('rank_sum', 0)
            row['rank_sum'] += rank
    selected = min(eligible_summaries, key=lambda r: (r['rank_sum'], r['k']))
    requested = study.get('selected_topic_count')
    if requested is not None:
        matches = [s for s in eligible_summaries if s['k'] == requested]
        if not matches:
            raise ValueError('selected_topic_count is unavailable in the fitted grid.')
        selected = min(matches, key=lambda r: r['rank_sum'])
    seed = study.get('selected_seed', 42)
    model = LdaModel(corpus=[corpus[i] for i in usable], id2word=dictionary, num_topics=selected['k'], random_state=seed,
                     alpha='auto', eta=selected['eta'], update_every=1, passes=study.get('lda_passes', 10),
                     iterations=study.get('lda_iterations', 100), minimum_probability=0.)
    topics, vectors = [], {}
    for k in range(selected['k']):
        words = [{'word': word, 'weight': float(weight)} for word, weight in model.show_topic(k, topn)]
        name = study.get('topic_names', {}).get(str(k+1), ' / '.join(w['word'] for w in words[:3]))
        topics.append({'topic_id': k+1, 'name': name, 'family': study.get('topic_families', {}).get(str(k+1)),
                       'words': words, 'label_status': 'researcher_supplied' if str(k+1) in study.get('topic_names', {}) else 'provisional_words'})
    for i, video in enumerate(videos):
        vectors[video['video_id']] = [float(p) for _, p in sorted(model.get_document_topics(corpus[i], minimum_probability=0.))] if corpus[i] else None
    representatives = []
    for topic in topics:
        best = sorted([v for v in videos if vectors[v['video_id']] is not None],
                      key=lambda v: vectors[v['video_id']][topic['topic_id']-1], reverse=True)[:5]
        representatives.extend({'topic_id': topic['topic_id'], 'video_id': v['video_id'], 'title': v['title'],
                                'weight': vectors[v['video_id']][topic['topic_id']-1],
                                'researcher1_label': '', 'researcher2_label': '', 'consensus_label': ''} for v in best)
    english = [i for i in usable if videos[i].get('original_language', '').lower().split('-')[0] == 'en']
    sensitivity = {'original_english_documents': len(english), 'status': 'insufficient_english_documents'}
    if len(english) >= max(4, selected['k']):
        english_model = LdaModel(corpus=[corpus[i] for i in english], id2word=dictionary, num_topics=selected['k'],
                                 random_state=seed, alpha='auto', eta=selected['eta'], passes=study.get('lda_passes', 10))
        left = [set(w for w, _ in model.show_topic(i, topn)) for i in range(selected['k'])]
        right = [set(w for w, _ in english_model.show_topic(i, topn)) for i in range(selected['k'])]
        sensitivity.update(status='fitted_same_vocabulary', matched_topword_jaccard=topic_similarity(left, right))
    output.mkdir(parents=True, exist_ok=True)
    model.save(str(output / 'lda.model'))
    dictionary.save(str(output / 'dictionary.gensim'))
    save(output / 'topics.json', topics)
    table(output / 'model_grid.csv', rows)
    table(output / 'model_selection.csv', summaries)
    table(output / 'representative_videos.csv', representatives)
    save(output / 'selection.json', {**selected, 'seed': seed, 'selection_status': 'researcher_selected_k' if requested else 'provisional_automatic_selection',
                                     'human_label_review_required': True, 'translation_sensitivity': sensitivity,
                                     'gibbs_sensitivity': 'not_implemented_do_not_claim',
                                     'perplexity_note': 'gensim variational held-out bound, not an exact likelihood',
                                     'empty_metadata_documents': len(videos)-len(usable)})
    table(output / 'video_topics.csv', [{'video_id': v['video_id'], 'topic_status': 'assigned' if vectors[v['video_id']] else 'empty_document',
                                        **{f'topic_{i+1}': p for i, p in enumerate(vectors[v['video_id']] or [])}} for v in videos])
    return topics, vectors
