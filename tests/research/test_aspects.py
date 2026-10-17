from pathlib import Path
from youtube_nlp.research.aspects import AspectExtractor

RULES=Path(__file__).resolve().parents[2]/'research/aspect_rules.json'


def test_multiaspect_negation_and_avatar_subtypes():
    extractor=AspectExtractor(RULES)
    result={r['aspect']:r for r in extractor.extract("Clear explanation but I don't understand. Your cute avatar keeps me company.")}
    assert result['A2']['polarity']=='positive'
    assert result['A4']['polarity']=='negative'
    assert result['A9']['subtypes']==['appearance','companionship']


def test_no_aspect_comment_retained_without_invented_targets():
    assert AspectExtractor(RULES).extract('Hello everybody!')==[]


def test_strict_dictionary_is_distinct_and_mentions_can_be_uninterpretable():
    extractor=AspectExtractor(RULES)
    assert extractor.extract('Your voice.')
    assert not extractor.extract('Your voice.',strict=True)
    assert extractor.extract('Your voice.')[0]['polarity'] is None
