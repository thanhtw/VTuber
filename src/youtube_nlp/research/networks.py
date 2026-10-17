"""RQ3 contextual networks, incidence exports, and explicit exploratory community nulls."""
from collections import Counter
from itertools import combinations
import math
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .statistics import associations, bh_adjust, PAIRS, fixed_margin_swap, bootstrap_ratios, finite, _totals
from .storage import table, save


def aspect_graph(rows, rules):
    graph = nx.Graph()
    for key, rule in rules['aspects'].items():
        graph.add_node(key, name=rule['name'], domain=rule['domain'])
    for row in rows:
        ratio = row['observed_expected_ratio']
        if ratio is not None and ratio > 1 and row['observed'] > 0:
            graph.add_edge(row['aspect_a'], row['aspect_b'], weight=math.log2(ratio),
                           observed=row['observed'], ratio=ratio)
    return graph


def partition(graph):
    if not graph.number_of_edges():
        return [{node} for node in graph], 0.
    groups = list(nx.community.greedy_modularity_communities(graph, weight='weight'))
    return groups, float(nx.community.modularity(graph, groups, weight='weight'))


def build_heterogeneous(videos, comments, topics, vectors, output, rules=None, channel_records=None):
    graph = nx.MultiDiGraph()
    for channel in channel_records or []:
        graph.add_node(channel['channel_id'], kind='channel',
                       **{key: value for key, value in channel.items() if key != 'channel_id' and value is not None})
    for key, rule in (rules or {}).get('aspects', {}).items():
        graph.add_node(key, kind='aspect', name=rule['name'], domain=rule['domain'])
    for topic in topics:
        graph.add_node(f"topic:{topic['topic_id']}", kind='topic', label=topic['name'], family=topic.get('family') or 'unassigned')
    for video in videos:
        vid, channel = video['video_id'], video['channel_id']
        graph.add_node(channel, kind='channel')
        graph.add_node(vid, kind='video', format=video['format'], language=video['original_language'],
                       **{key: value for key, value in video.items() if key in {'age_days','views','likes','reported_comments','likes_missing','views_missing','duration','category','log1p_views_per_day','likes_per_view','comments_per_view'} and value is not None})
        graph.add_edge(channel, vid, relation='publishes', weight=1.)
        for kind, value in [('format', video['format']), ('language', video['original_language'])]:
            node = f'{kind}:{value}'
            graph.add_node(node, kind=kind)
            graph.add_edge(vid, node, relation=f'has_{kind}', weight=1.)
        for k, probability in enumerate(vectors.get(vid) or []):
            graph.add_edge(vid, f'topic:{k+1}', relation='covers', weight=float(probability))
    evidence = []
    for comment in comments:
        cid = comment['comment_id']
        graph.add_node(cid, kind='comment', polarity=comment['polarity'], language=comment['source_language'])
        graph.add_edge(comment['video_id'], cid, relation='contains', weight=1.)
        if comment.get('parent_id'):
            graph.add_edge(cid, comment['parent_id'], relation='reply_to', weight=1.)
        for aspect in comment['aspects']:
            graph.add_node(aspect['aspect'], kind='aspect', name=aspect['name'], domain=aspect['domain'])
            graph.add_edge(cid, aspect['aspect'], relation='mentions', weight=1., polarity=aspect['polarity'] or 'uninterpretable')
        for a, b in combinations(comment['aspects'], 2):
            evidence.append({'comment_id': cid, 'video_id': comment['video_id'], 'aspect_a': a['aspect'], 'aspect_b': b['aspect']})
    nx.write_graphml(graph, output / 'heterogeneous.graphml')
    table(output / 'nodes.csv', [{'id': node, **attrs} for node, attrs in graph.nodes(data=True)])
    table(output / 'edges.csv', [{'source': a, 'target': b, **attrs} for a, b, attrs in graph.edges(data=True)])
    table(output / 'pair_evidence.csv', evidence, ['comment_id', 'video_id', 'aspect_a', 'aspect_b'])


def analyze_networks(videos, comments, topics, vectors, extractor, study, output, channel_records=None):
    output.mkdir(parents=True, exist_ok=True)
    build_heterogeneous(videos, comments, topics, vectors, output, extractor.rules, channel_records)
    aspects = [f'A{i}' for i in range(1, 10)]
    x = np.array([[int(a in {r['aspect'] for r in c['aspects']}) for a in aspects] for c in comments], dtype=int).reshape((-1, 9))
    strict = np.array([[int(a in {r['aspect'] for r in c['strict_aspects']}) for a in aspects] for c in comments], dtype=int).reshape((-1, 9))
    channels = [c['channel_id'] for c in comments]
    table(output / 'comment_aspect_incidence.csv', [{'comment_id': c['comment_id'], 'video_id': c['video_id'],
                                                   'channel_id': c['channel_id'], **dict(zip(aspects, map(int, row)))} for c, row in zip(comments, x)],
          ['comment_id', 'video_id', 'channel_id', *aspects])
    if not comments:
        save(output / 'status.json', {'status': 'no_usable_comments'})
        return
    by_video = {v['video_id']: v for v in videos}
    weights = {'all': np.ones(len(comments))}
    for fmt in sorted({v['format'] for v in videos}):
        weights[f'format:{fmt}'] = np.array([float(by_video[c['video_id']]['format'] == fmt) for c in comments])
    for lang in sorted({c['source_language'] for c in comments}):
        weights[f'language:{lang}'] = np.array([float(c['source_language'] == lang) for c in comments])
    for topic in topics:
        weights[f"topic:{topic['topic_id']}"] = np.array([(vectors.get(c['video_id']) or [0.]*len(topics))[topic['topic_id']-1] for c in comments])
    families = {t['family'] for t in topics if t.get('family')}
    for family in families:
        indices = [t['topic_id']-1 for t in topics if t.get('family') == family]
        weights[f'family:{family}'] = np.array([sum((vectors.get(c['video_id']) or [0.]*len(topics))[i] for i in indices) for c in comments])
    counts = Counter(channels)
    largest = max(counts, key=counts.get)
    weights['sensitivity:exclude_largest_channel'] = np.array([float(c != largest) for c in channels])
    weights['sensitivity:original_english'] = np.array([float(c['source_language'].lower().split('-')[0] == 'en') for c in comments])
    weights['sensitivity:translated_non_english'] = np.array([float(c['source_language'].lower().split('-')[0] not in {'en', 'und', 'unknown'}) for c in comments])
    video_counts = Counter(c['video_id'] for c in comments)
    weights['sensitivity:equal_video_weight'] = np.array([1/video_counts[c['video_id']] for c in comments])
    all_rows = []
    for label, w in weights.items():
        mask = w > 0
        if not mask.any():
            continue
        subset_channels = list(np.asarray(channels)[mask])
        rows = associations(x[mask], subset_channels, w[mask], study.get('bootstrap_samples', 1000), study.get('permutations', 1000), study.get('seed', 42))
        all_rows.extend({'context': label, **row} for row in rows)
    strict_rows = associations(strict, channels, bootstrap=study.get('bootstrap_samples', 1000), permutations=study.get('permutations', 1000), seed=study.get('seed', 42))
    all_rows.extend({'context': 'sensitivity:strict_dictionary', **row} for row in strict_rows)
    fixed_rows = associations(x, channels, bootstrap=study.get('bootstrap_samples', 1000), permutations=study.get('fixed_margin_permutations', 200), seed=study.get('seed', 42), null='fixed_margin_within_channel')
    all_rows.extend({'context': 'sensitivity:fixed_comment_degree', **row} for row in fixed_rows)
    for row, q in zip(all_rows, bh_adjust([r['p_two_sided'] for r in all_rows])):
        row['q_across_reported_networks'] = q
    table(output / 'aspect_associations.csv', all_rows)
    global_rows = [r for r in all_rows if r['context'] == 'all']
    graph = aspect_graph(global_rows, extractor.rules)
    communities, modularity = partition(graph)
    nx.write_graphml(graph, output / 'aspect_network.graphml')
    # Independent fixed-margin randomizations, preserving each comment's aspect count.
    rng = np.random.default_rng(study.get('seed', 42))
    groups = [np.flatnonzero(np.asarray(channels) == key) for key in sorted(set(channels))]
    null_modularities, accepted = [], 0
    for _ in range(study.get('community_null_samples', 100)):
        randomized, moved = fixed_margin_swap(x, groups, rng, max(100, 5*len(x)))
        accepted += moved
        total, marginal, joint, expected, ratios = _totals(randomized, np.ones(len(x)))
        null_rows = [{'aspect_a': f'A{a+1}', 'aspect_b': f'A{b+1}', 'observed': float(joint[j]),
                      'observed_expected_ratio': finite(ratios[j])} for j, (a, b) in enumerate(PAIRS)]
        null_modularities.append(partition(aspect_graph(null_rows, extractor.rules))[1])
    # Channel bootstrap community coassignment probabilities.
    coassignment = np.zeros((9, 9))
    bootstrap_runs = study.get('community_bootstrap_samples', 100)
    unique = sorted(set(channels))
    for _ in range(bootstrap_runs if len(unique) >= 2 else 0):
        sample = Counter(rng.choice(unique, size=len(unique), replace=True))
        w = np.array([sample[c] for c in channels], dtype=float)
        _, _, joint, _, ratios = _totals(x, w)
        rows = [{'aspect_a': f'A{a+1}', 'aspect_b': f'A{b+1}', 'observed': float(joint[j]), 'observed_expected_ratio': finite(ratios[j])} for j, (a, b) in enumerate(PAIRS)]
        for group in partition(aspect_graph(rows, extractor.rules))[0]:
            for a in group:
                for b in group:
                    coassignment[int(a[1:])-1, int(b[1:])-1] += 1/bootstrap_runs
    table(output / 'community_stability.csv', [{'aspect_a': a, 'aspect_b': b,
                                                'channel_bootstrap_coassignment': float(coassignment[i, j]) if len(unique) >= 2 else None}
                                               for i, a in enumerate(aspects) for j, b in enumerate(aspects)])
    save(output / 'communities.json', {'communities': [sorted(g) for g in communities], 'modularity': modularity,
          'null_modularities': null_modularities, 'fixed_margin_swaps_accepted': accepted,
          'p_modularity_enrichment': (1+sum(v >= modularity for v in null_modularities))/(len(null_modularities)+1) if accepted else None,
          'null_mixing_note': 'Finite checkerboard randomization; inspect movement and sensitivity, not proof of exact uniform sampling.',
          'weight': 'positive log2(observed/independence_expected); depleted pairs remain in association CSV',
          'status': 'exploratory_dictionary_indicators_not_validated_constructs', 'gnn': 'omitted_no_independent_prediction_target'})
    # Direct contrasts share bootstrap channel multiplicities, rather than comparing CI overlap.
    contrasts = []
    for prefix in ('format:', 'topic:', 'language:', 'family:'):
        for left, right in combinations([k for k in weights if k.startswith(prefix)], 2):
            a = bootstrap_ratios(x, channels, weights[left], study.get('bootstrap_samples', 1000), study.get('seed', 42))
            b = bootstrap_ratios(x, channels, weights[right], study.get('bootstrap_samples', 1000), study.get('seed', 42))
            point_a = _totals(x, weights[left])[4]
            point_b = _totals(x, weights[right])[4]
            for j, (i, k) in enumerate(PAIRS):
                valid = (a[:, j]-b[:, j]) if a is not None else np.array([])
                valid = valid[np.isfinite(valid)]
                ci = np.quantile(valid, [.025, .975]) if len(valid) else [np.nan, np.nan]
                contrasts.append({'left': left, 'right': right, 'aspect_a': f'A{i+1}', 'aspect_b': f'A{k+1}',
                                  'ratio_difference': finite(point_a[j]-point_b[j]), 'ci_low': finite(ci[0]), 'ci_high': finite(ci[1]),
                                  'status': 'exploratory_unadjusted_bootstrap_contrast'})
    table(output / 'context_contrasts.csv', contrasts, ['left', 'right', 'aspect_a', 'aspect_b', 'ratio_difference', 'ci_low', 'ci_high', 'status'])
    table(output / 'video_aspect_prevalence.csv', [{'video_id': vid, 'comments': sum(c['video_id'] == vid for c in comments),
              **{a: float(x[np.array([c['video_id'] == vid for c in comments]), i].mean()) for i, a in enumerate(aspects)}}
              for vid in sorted({c['video_id'] for c in comments})])
    fig, axis = plt.subplots(figsize=(9, 7))
    pos = nx.circular_layout(graph)
    nx.draw_networkx(graph, pos, ax=axis, node_color=['#71a6d2' if graph.nodes[n]['domain'] == 'CoI' else '#f4ba68' if graph.nodes[n]['domain'] == 'CTML' else '#a391cf' for n in graph],
                     width=[max(.5, d['weight']*2) for _, _, d in graph.edges(data=True)])
    axis.set_title('Exploratory aspect associations (positive log2 observed/expected)')
    axis.axis('off')
    fig.tight_layout()
    fig.savefig(output / 'aspect_network.png', dpi=180)
    plt.close(fig)
    save(output / 'status.json', {'comments': len(comments), 'comments_with_multiple_aspects': int((x.sum(axis=1) >= 2).sum()),
                                 'all_comment_denominator': True, 'largest_channel_excluded_in_sensitivity': largest,
                                 'pair_tests': 36, 'interval_note': 'Channel percentile bootstrap; missing if fewer than two channels.'})
