import pytest
from pathlib import Path
from youtube_nlp.config import Config
from youtube_nlp.research.settings import Study
from youtube_nlp.research.topics import chapter_titles, topic_similarity, fit_topics


def test_chapters_and_label_permutation_matching():
    assert chapter_titles('00:00 Introduction\n01:12 Explanation\nJust text') == ['Introduction','Explanation']
    assert topic_similarity([{'a','b'},{'c','d'}],[{'c','d'},{'a','b'}]) == 1


def test_real_gensim_grid_probabilities_and_heldout_outputs(tmp_path):
    pytest.importorskip('gensim')
    from nltk.stem import WordNetLemmatizer
    try:
        WordNetLemmatizer().lemmatize('lessons')
    except LookupError:
        pytest.skip('wordnet unavailable')
    videos=[{'video_id':f'v{i}','title':'Synthetic metadata', 'metadata_text_en':text,'original_language':'en'} for i,text in enumerate([
        'python code function variable programming software', 'python software algorithm function code',
        'programming function variable python code', 'algorithm software code python variable',
        'music singing concert melody rhythm performance','singing melody rhythm music performance',
        'concert music melody singing rhythm','rhythm performance music concert singing'])]
    study=Study(Config('', 'test', 1, 1, 'en', 2, 3, 42, tmp_path),tmp_path,
                {'topic_counts':[2], 'seeds':[1,2], 'eta_grid':[.1], 'selected_seed':1, 'lda_passes':2,'lda_iterations':20,
                 'min_document_frequency':1,'max_document_fraction':1.,'topic_top_words':3})
    topics,vectors=fit_topics(videos,study,tmp_path/'rq1')
    assert len(topics)==2
    assert all(sum(p)==pytest.approx(1,abs=1e-6) for p in vectors.values())
    assert (tmp_path/'rq1/model_grid.csv').exists()
    assert (tmp_path/'rq1/selection.json').exists()
