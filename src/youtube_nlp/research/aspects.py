"""Auditable multilabel dictionary extraction; co-mentions never imply psychological constructs."""
import re
from pathlib import Path
from .storage import read_json


class AspectExtractor:
    def __init__(self, rules):
        self.rules = read_json(Path(rules))
        if set(self.rules.get("aspects", {})) != {f"A{i}" for i in range(1, 10)}:
            raise ValueError("Aspect rules must define A1 through A9.")
        self.version = self.rules["version"]
        for rule in self.rules['aspects'].values():
            for name in ("patterns", "strict_patterns"):
                for pattern in rule[name]:
                    re.compile(pattern)

    def extract(self, text, strict=False):
        # Split contrastive clauses to avoid copying one target's polarity onto another.
        clauses = re.split(r"(?<=[.!?;])\s+|\b(?:but|however|whereas)\b", text, flags=re.I)
        found = {}
        for clause in clauses:
            lower = clause.lower().replace("’", "'")
            matches = {key: [pattern for pattern in rule['strict_patterns' if strict else 'patterns']
                             if re.search(pattern, lower)] for key, rule in self.rules['aspects'].items()}
            for key, patterns in matches.items():
                if not patterns:
                    continue
                rule = self.rules['aspects'][key]
                positive = [phrase for phrase in rule.get('positive', []) if re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', lower)]
                negative = [phrase for phrase in rule.get('negative', []) if re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', lower)]
                # Negative understanding dominates positive substrings: 'do not understand'.
                if negative:
                    polarity = 'negative' if not positive or key == 'A4' else 'mixed'
                elif positive:
                    # Nearby explicit negation reverses positive rules, preserving match evidence.
                    negated = any(re.search(r"\b(?:not|never|isn't|wasn't|don't|didn't)\s+(?:\w+\s+){0,2}" + re.escape(p), lower) for p in positive)
                    polarity = 'negative' if negated else 'positive'
                else:
                    polarity = None  # A mention alone does not establish its evaluative polarity.
                subtypes = [name for name, pats in rule.get('subtypes', {}).items() if any(re.search(p, lower) for p in pats)]
                item = found.setdefault(key, {"aspect": key, "name": rule['name'], "domain": rule['domain'],
                                             "polarity": None, "evidence": [], "matched_rules": [], "subtypes": [], "rules_version": self.version})
                item['evidence'].append(clause.strip())
                item['matched_rules'].extend(patterns)
                item['subtypes'] = sorted(set(item['subtypes']) | set(subtypes))
                if polarity:
                    item['polarity'] = polarity if item['polarity'] in (None, polarity) else 'mixed'
        return list(found.values())
