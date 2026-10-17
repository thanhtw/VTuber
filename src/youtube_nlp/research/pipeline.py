"""Manuscript analysis with provenance, coded IDs, audit templates, and no synthetic findings."""
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import math
import secrets
import uuid
from pathlib import Path
import numpy as np
from .storage import read_json, save, table, csv_rows, utcnow, digest
from .topics import chapter_titles, clean_metadata, fit_topics
from .sentiment import Sentiment
from .screening import require_teaching_evidence
from .aspects import AspectExtractor
from .statistics import proportion
from .networks import analyze_networks
from ..translation import translate_comments

log = logging.getLogger(__name__)


def number(stats, key):
    value = stats.get(key)
    return int(value) if value is not None else None


def elapsed_days(published, collected):
    if not published:
        return None
    return max(1., (datetime.fromisoformat(collected.replace('Z', '+00:00')) - datetime.fromisoformat(published.replace('Z', '+00:00'))).total_seconds()/86400)


def normalize(dataset):
    now = dataset['collected_at']
    videos = []
    for raw in dataset['videos']:
        snippet, stats, details = raw['snippet'], raw.get('statistics', {}), raw.get('contentDetails', {})
        age = elapsed_days(snippet.get('publishedAt'), raw.get('collected_at', now))
        views, likes, comments = [number(stats, k) for k in ('viewCount', 'likeCount', 'commentCount')]
        description = snippet.get('description', '')
        video = {'video_id': raw['id'], 'channel_id': snippet['channelId'], 'title': snippet.get('title', ''),
                 'description': description, 'tags': snippet.get('tags', []), 'chapter_titles': chapter_titles(description),
                 'published_at': snippet.get('publishedAt'), 'collected_at': raw.get('collected_at', now),
                 'duration': details.get('duration'), 'category': snippet.get('categoryId'), 'caption_available': details.get('caption'),
                 'has_livestream_metadata': bool(raw.get('liveStreamingDetails')), 'format': raw['format'],
                 'declared_language': snippet.get('defaultAudioLanguage', snippet.get('defaultLanguage')),
                 'teaching_eligibility_evidence': raw.get('teaching_eligibility_evidence', {}),
                 'age_days': age, 'views': views, 'likes': likes, 'reported_comments': comments,
                 'likes_missing': likes is None, 'views_missing': views is None,
                 'likes_per_view': likes/views if likes is not None and views else None,
                 'comments_per_view': comments/views if comments is not None and views else None}
        for name, value in [('views', views), ('likes', likes), ('comments', comments)]:
            video[f'{name}_per_day'] = value/age if value is not None and age else None
            video[f'log1p_{name}_per_day'] = math.log1p(value/age) if value is not None and age else None
        videos.append(video)
    return videos


def code_ids(videos, comments, root):
    salt_path = root / 'private_id_salt.json'
    salt = read_json(salt_path, {}).get('salt') or secrets.token_hex(32)
    save(salt_path, {'salt': salt})
    mapping = {}
    def code(kind, identifier):
        if identifier is None:
            return None
        key = f'{kind}:{identifier}'
        mapping[key] = f'{kind}_' + hashlib.sha256((salt + key).encode()).hexdigest()[:20]
        return mapping[key]
    for video in videos:
        video['video_id'] = code('video', video['video_id'])
        video['channel_id'] = code('channel', video['channel_id'])
    for comment in comments:
        comment['comment_id'] = code('comment', comment['comment_id'])
        comment['video_id'] = code('video', comment['video_id'])
        comment['channel_id'] = code('channel', comment['channel_id'])
        comment['parent_id'] = code('comment', comment.get('parent_id'))
    save(root / 'private_identifier_map.json', mapping)
    return mapping


def apply_translation_reviews(comments, study):
    reviews = {r['comment_id']: r for r in csv_rows(study.root / 'translation_reviews.csv')}
    applied = 0
    for comment in comments:
        row = reviews.get(comment['comment_id'])
        if row and row.get('reviewed_text_en', '').strip() and row.get('reviewer1') and row.get('reviewer2') and row['reviewer1'] != row['reviewer2']:
            if row.get('original_sha256') != hashlib.sha256(comment['text'].encode()).hexdigest():
                raise ValueError('Translation review text hash does not match the collected original.')
            comment['text_en'] = row['reviewed_text_en']
            comment['translation_reviewed'] = True
            applied += 1
    return applied


def audit_templates(comments, extractor, study, output):
    rng = np.random.default_rng(study.get('seed', 42))
    sampled = []
    by_language = defaultdict(list)
    for comment in comments:
        by_language[comment['source_language']].append(comment)
    for language, candidates in by_language.items():
        selected = rng.choice(len(candidates), min(study.get('audit_sample_per_language', 30), len(candidates)), replace=False)
        for i in selected:
            comment = candidates[int(i)]
            sampled.append({'comment_id': comment['comment_id'], 'source_language': language,
                            'original_sha256': hashlib.sha256(comment['text'].encode()).hexdigest(),
                            'text': comment['text'], 'text_en': comment['text_en'], 'reviewer1': '', 'reviewer2': '',
                            'meaning_preserved1': '', 'meaning_preserved2': '', 'reviewed_text_en': '', 'notes': ''})
    table(output / 'translation_review_sample.csv', sampled)
    rows = []
    for aspect in extractor.rules['aspects']:
        candidates = [c for c in comments if aspect in {a['aspect'] for a in c['aspects']}]
        selected = rng.choice(len(candidates), min(study.get('aspect_audit_per_aspect', 30), len(candidates)), replace=False)
        for i in selected:
            comment = candidates[int(i)]
            rows.append({'comment_id': comment['comment_id'], 'aspect': aspect, 'text': comment['text'], 'text_en': comment['text_en'],
                         'source_language': comment['source_language'], 'rules_sha256': digest(extractor.rules),
                         'text_en_sha256': hashlib.sha256(comment['text_en'].encode()).hexdigest(), 'reviewer1': '', 'reviewer2': '', 'correct1': '', 'correct2': '',
                         'consensus_correct': '', 'notes': ''})
    table(output / 'aspect_precision_audit_sample.csv', rows)
    audits = csv_rows(study.root / 'aspect_precision_audit.csv')
    predictions = {(c['comment_id'], a['aspect']): hashlib.sha256(c['text_en'].encode()).hexdigest() for c in comments for a in c['aspects']}
    estimates = []
    for aspect in extractor.rules['aspects']:
        valid = [r for r in audits if r.get('aspect') == aspect and r.get('rules_sha256') == digest(extractor.rules)
                 and predictions.get((r.get('comment_id'), aspect)) == r.get('text_en_sha256') and r.get('reviewer1') and r.get('reviewer2')
                 and r['reviewer1'] != r['reviewer2'] and r.get('correct1') in {'0','1'} and r.get('correct2') in {'0','1'}
                 and (r.get('consensus_correct') in {'0','1'} or r['correct1'] == r['correct2'])]
        valid = list({r['comment_id']: r for r in valid}.values())
        true = sum(int(r.get('consensus_correct') or r['correct1']) for r in valid)
        n = len(valid)
        precision = true/n if n else None
        if n:
            z = 1.96
            center = (precision+z*z/(2*n))/(1+z*z/n)
            half = z*math.sqrt(precision*(1-precision)/n+z*z/(4*n*n))/(1+z*z/n)
            lo, hi = center-half, center+half
        else:
            lo, hi = None, None
        estimates.append({'aspect': aspect, 'audited_predictions': n, 'correct': true, 'precision': precision,
                          'wilson_low': lo, 'wilson_high': hi, 'recall': None,
                          'status': 'precision_only_sample_audit' if n else 'not_independently_audited'})
    table(output / 'aspect_precision_audit_results.csv', estimates)
    return {'translation_review_sample': len(sampled), 'aspect_precision_sample': len(rows),
            'audited_aspect_predictions': sum(r['audited_predictions'] for r in estimates),
            'note': 'Audit samples and blank reviewer fields do not constitute human validation.'}


def rq2_tables(videos, comments, topics, vectors, study, output):
    usable = [c for c in comments if c.get('polarity') in {'positive', 'negative', 'neutral'}]
    channels = [c['channel_id'] for c in usable]
    contexts = {'all': np.ones(len(usable))}
    for topic in topics:
        contexts[f"topic:{topic['topic_id']}"] = np.array([(vectors.get(c['video_id']) or [0.]*len(topics))[topic['topic_id']-1] for c in usable])
    for family in {t['family'] for t in topics if t.get('family')}:
        indices = [t['topic_id']-1 for t in topics if t.get('family') == family]
        contexts[f'family:{family}'] = np.array([sum((vectors.get(c['video_id']) or [0.]*len(topics))[i] for i in indices) for c in usable])
    by_video = {v['video_id']: v for v in videos}
    for fmt in sorted({v['format'] for v in videos}):
        contexts[f'format:{fmt}'] = np.array([float(by_video[c['video_id']]['format'] == fmt) for c in usable])
    for label, predicate in [('original_english', lambda c: c['source_language'].lower().split('-')[0] == 'en'),
                              ('translated_non_english', lambda c: c['source_language'].lower().split('-')[0] not in {'en','unknown','und'})]:
        contexts[f'sensitivity:{label}'] = np.array([float(predicate(c)) for c in usable])
    if channels:
        largest = Counter(channels).most_common(1)[0][0]
        contexts['sensitivity:exclude_largest_channel'] = np.array([float(c['channel_id'] != largest) for c in usable])
    overall, aspect_rows = [], []
    for label, weights in contexts.items():
        for polarity in ['positive', 'negative', 'neutral']:
            result = proportion([float(c['polarity'] == polarity) for c in usable], weights, channels, study.get('bootstrap_samples',1000), study.get('seed',42))
            overall.append({'context': label, 'polarity': polarity, **result})
        for aspect in [f'A{i}' for i in range(1,10)]:
            mentions = [next((a for a in c['aspects'] if a['aspect'] == aspect), None) for c in usable]
            mention_weights = weights*np.array([m is not None for m in mentions])
            result = proportion([float(m is not None) for m in mentions], weights, channels, study.get('bootstrap_samples',1000), study.get('seed',42))
            aspect_rows.append({'context': label, 'aspect': aspect, 'measure': 'mention_prevalence', **result})
            for polarity in ['positive', 'negative', 'mixed', 'neutral', 'uninterpretable']:
                result = proportion([float((m or {}).get('polarity') == (None if polarity == 'uninterpretable' else polarity)) for m in mentions],
                                    mention_weights, channels, study.get('bootstrap_samples',1000), study.get('seed',42))
                aspect_rows.append({'context': label, 'aspect': aspect, 'measure': f'{polarity}_conditional_on_mention', **result})
    table(output / 'overall_sentiment_by_context.csv', overall)
    table(output / 'aspect_prevalence_and_polarity.csv', aspect_rows)
    table(output / 'a9_subtypes.csv', [{'subtype': subtype,
          **proportion([float(any(a['aspect'] == 'A9' and subtype in a['subtypes'] for a in c['aspects'])) for c in usable], np.ones(len(usable)), channels, study.get('bootstrap_samples',1000), study.get('seed',42))}
          for subtype in ['appearance', 'companionship', 'attachment']])
    video_rows = []
    for video in videos:
        rows = [c for c in comments if c['video_id'] == video['video_id']]
        reply_count = sum(bool(c.get('is_reply')) for c in rows)
        top = [c for c in rows if not c.get('is_reply')]
        creator_parents = {c['parent_id'] for c in rows if c.get('is_reply') and c.get('is_creator')}
        video_rows.append({'video_id': video['video_id'], 'retrieved_comments': len(rows), 'top_level_comments': len(top),
                           'replies': reply_count, 'reply_ratio': reply_count/len(rows) if rows else None,
                           'creator_reply_rate': len(creator_parents)/len(top) if top else None})
    table(output / 'video_response_indicators.csv', video_rows)


def run_analysis(study, scorer=None, topic_fitter=None, translator=None):
    study.prepare()
    dataset = read_json(study.root / 'dataset.json')
    if not isinstance(dataset, dict) or not all(k in dataset for k in ('channels', 'videos', 'comments', 'collected_at')):
        raise ValueError('No research dataset. Run discovery, complete screening, then run collect.')
    if not dataset.get('collection_complete'):
        raise ValueError('Research collection is incomplete. Rerun collect to finish before analysis.')
    if not dataset['videos']:
        raise ValueError('Research dataset contains no eligible videos.')
    for video in dataset['videos']:
        require_teaching_evidence({'id': video['id'], **video.get('teaching_eligibility_evidence', {})})
    if not study.api.openai_api_key and translator is None:
        raise ValueError('Set OPENAI_API_KEY in .env. Metadata and comments must be translated before analysis.')
    if len({v['id'] for v in dataset['videos']}) != len(dataset['videos']) or len({c['comment_id'] for c in dataset['comments']}) != len(dataset['comments']):
        raise ValueError('Duplicate video/comment identifiers in research dataset.')
    video_ids = {v['id'] for v in dataset['videos']}
    if any(c['video_id'] not in video_ids for c in dataset['comments']):
        raise ValueError('A comment is not linked to an eligible video.')
    owners = {v['id']: v['snippet']['channelId'] for v in dataset['videos']}
    if any(c.get('channel_id') != owners[c['video_id']] for c in dataset['comments']):
        raise ValueError('Comment channel ID disagrees with its source video.')
    # Resolve local dependencies before incurring translation API usage.
    scorer = scorer or Sentiment(study)
    if topic_fitter is None:
        try:
            import gensim
            from nltk.stem import WordNetLemmatizer
            WordNetLemmatizer().lemmatize('lessons')
        except (ImportError, LookupError):
            raise ValueError('Install research dependencies and run: python -m nltk.downloader wordnet omw-1.4') from None
    videos = normalize(dataset)
    translator = translator or translate_comments
    translation_config = replace(study.api, output_dir=study.root, translate_comments=True)
    metadata = [{'comment_id': v['video_id'], 'text': text} for v, text in zip(videos, clean_metadata(videos, study))]
    translated_metadata = translator(metadata, translation_config)
    for video, translated in zip(videos, translated_metadata):
        video['metadata_text_en'] = translated['text_en']
        video['metadata_detected_language'] = translated['source_language']
        video['original_language'] = video['declared_language'] or translated['source_language']
    comments = translator([dict(c) for c in dataset['comments']], translation_config)
    if len(comments) != len(dataset['comments']) or len(translated_metadata) != len(videos):
        raise ValueError('Translation did not return one record per source record.')
    if any(c['text'].strip() and not c.get('text_en','').strip() for c in comments):
        raise ValueError('A nonempty comment lacks an English translation; analysis stopped.')
    mapping = code_ids(videos, comments, study.root)
    overrides = apply_translation_reviews(comments, study)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    output = study.root / 'analyses' / run_id
    output.mkdir(parents=True, mode=0o700)
    save(output / 'run_status.json', {'status': 'started', 'started_at': utcnow()})
    topics, vectors = (topic_fitter or fit_topics)(videos, study, output / 'rq1')
    rules_path = Path(study.get('aspect_rules_path'))
    extractor = AspectExtractor(rules_path)
    usable = [c for c in comments if c['text'].strip() and c['text_en'].strip()]
    scores = scorer.score([c['text_en'] for c in usable])
    if len(scores) != len(usable):
        raise ValueError('Sentiment prediction count does not match usable comments.')
    by_comment = {c['comment_id']: s for c, s in zip(usable, scores)}
    for comment in comments:
        score = by_comment.get(comment['comment_id'])
        comment.update(score or {'polarity': None})
        comment['usable_text'] = score is not None
        comment['aspects'] = extractor.extract(comment['text_en']) if score else []
        comment['strict_aspects'] = extractor.extract(comment['text_en'], strict=True) if score else []
    save(output / 'analyzed_dataset.json', {'videos': videos, 'comments': comments, 'topics': topics, 'topic_vectors': vectors})
    table(output / 'comments.csv', [{k:v for k,v in c.items() if k not in {'strict_aspects'}} for c in comments])
    table(output / 'video_metadata.csv', videos)
    channel_records = []
    for raw in dataset['channels']:
        stats, snippet = raw.get('statistics', {}), raw.get('snippet', {})
        coded = mapping.get(f"channel:{raw['id']}")
        if not coded:
            continue
        subscribers = None if stats.get('hiddenSubscriberCount') else number(stats, 'subscriberCount')
        channel_records.append({'channel_id': coded, 'created_at': snippet.get('publishedAt'),
                                'channel_age_days': elapsed_days(snippet.get('publishedAt'), dataset['collected_at']),
                                'declared_country': snippet.get('country'), 'declared_language': snippet.get('defaultLanguage'),
                                'subscribers': subscribers, 'subscribers_missing': subscribers is None,
                                'log1p_subscribers': math.log1p(subscribers) if subscribers is not None else None,
                                'uploads': number(stats, 'videoCount'),
                                'included_videos': sum(v['channel_id'] == coded for v in videos),
                                'screened_lesson_share': None, 'lesson_share_status': 'not_estimated_until_all_uploads_are_screened'})
    table(output / 'channels.csv', channel_records)
    rq2_tables(videos, comments, topics, vectors, study, output)
    analyze_networks(videos, usable, topics, vectors, extractor, study, output / 'rq3', channel_records=channel_records)
    audits = audit_templates(usable, extractor, study, output)
    summary = {'channels': len({v['channel_id'] for v in videos}), 'eligible_accessible_videos': len(videos),
               'comments_total': len(comments), 'usable_comments': len(usable), 'unusable_comments': len(comments)-len(usable),
               'comments_excluded_for_learning_relevance': 0, 'top_level_comments': sum(not c.get('is_reply') for c in comments),
               'replies': sum(bool(c.get('is_reply')) for c in comments),
               'videos_with_comments': len({c['video_id'] for c in comments}),
               'comments_with_aspects': sum(bool(c['aspects']) for c in comments),
               'comments_without_aspects': sum(not c['aspects'] for c in comments),
               'coverage_issues': [{k: mapping.get(f"{'video' if k == 'video_id' else 'comment'}:{v}", v) if k in {'video_id','parent_id'} else v for k,v in issue.items()} for issue in dataset.get('coverage_issues', [])], 'translation_overrides_applied': overrides,
               'human_validation': audits, 'screening': read_json(study.root / 'screening_summary.json', {})}
    registry = study.root / 'query_registry.jsonl'
    query_runs = [json.loads(line) for line in registry.read_text(encoding='utf-8').splitlines() if line.strip()] if registry.exists() else []
    summary['observed_search_dates_utc'] = sorted({r['time'][:10] for r in query_runs if r.get('resource') == 'search' and r.get('status') == 'ok'})
    summary['repeated_search_date_check'] = 'at_least_two_observed_dates' if len(summary['observed_search_dates_utc']) >= 2 else 'not_demonstrated'
    summary['query_api_calls'] = len([r for r in query_runs if r.get('resource') == 'search'])
    save(output / 'sample_overview.json', summary)
    table(output / 'table5_sample_overview.csv', [{'indicator': k, 'value': v} for k,v in summary.items() if isinstance(v, (int,float))])
    table(output / 'table6_topic_summary.csv', [{'topic_id': t['topic_id'], 'name': t['name'], 'family': t.get('family'),
            'effective_video_count': sum((vectors.get(v['video_id']) or [0.]*len(topics))[t['topic_id']-1] for v in videos),
            'denominator_assigned_videos': sum(vectors.get(v['video_id']) is not None for v in videos)} for t in topics])
    packages = {}
    for package in ['gensim', 'numpy', 'scipy', 'nltk', 'networkx', 'transformers', 'torch']:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = 'not_installed'
    code_hash = digest({str(p.relative_to(Path(__file__).parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in Path(__file__).parent.parent.rglob('*.py')})
    manifest = {'run_id': run_id, 'created_at': utcnow(), 'dataset_sha256': digest(dataset), 'study_settings': study.options,
                'study_settings_sha256': digest(study.options), 'code_sha256': code_hash, 'packages': packages,
                'sentiment_model': scorer.identifier, 'sentiment_resolved_revision': scorer.revision,
                'translation_model': study.api.translation_model, 'aspect_rules_sha256': digest(extractor.rules),
                'status': 'provisional_empirical_outputs_require_human_audits',
                'method_deviations': ['No collapsed Gibbs sampler robustness test implemented.',
                                     'Topic count/rank and automatic names need independent researcher review.',
                                     'Aspect rules are a pilot dictionary requiring independent precision audit.',
                                     'No automatic inference of latent presence, learning achievement, or causation.',
                                     'No GNN: no independent prediction target has been defined.']}
    if scorer.backend != 'transformer':
        manifest['method_deviations'].append('VADER exploratory backend used instead of the manuscript pretrained transformer.')
    save(output / 'manifest.json', manifest)
    save(output / 'run_status.json', {'status': 'complete', 'completed_at': utcnow()})
    save(study.root / 'latest_analysis.json', {'path': str(output), 'run_id': run_id})
    report_text = f'''# Empirical output inventory — {run_id}

These outputs are computed from collected records, never copied from manuscript examples.

- Sample: {summary['channels']} channels; {len(videos)} videos; {len(comments)} comments/replies; {len(usable)} usable texts.
- RQ1: rq1/model_grid.csv, model_selection.csv, representative_videos.csv, topics.json and video_topics.csv.
- RQ2: overall_sentiment_by_context.csv and aspect_prevalence_and_polarity.csv. Every proportion includes numerator, denominator and channel bootstrap interval.
- RQ3: rq3/aspect_associations.csv (all 36 pairs), context_contrasts.csv, incidence, graph files, pair evidence and community null/stability results.
- Original and English text: comments.csv and analyzed_dataset.json (restricted research data).
- Independent checks: translation_review_sample.csv and aspect_precision_audit_sample.csv are blank review templates, not completed validation.

Do not present these as final manuscript findings before reviewing screening, translation, aspects, topic labels, channel counts and model diagnostics. Missing confidence intervals indicate insufficient channel replication, not zero uncertainty. Consult manifest.json for exact model versions and departures from the manuscript.
'''
    (output / 'RESULTS_README.md').write_text(report_text, encoding='utf-8')
    log.info('Research analysis saved to %s', output)
    return output
