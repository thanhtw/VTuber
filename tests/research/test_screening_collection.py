from pathlib import Path
from unittest.mock import Mock
import pytest
from youtube_nlp.research.screening import decision, update_template, report
from youtube_nlp.research.collection import collect_comments, pages, discover, collect
from youtube_nlp.research.storage import save, table, read_json
from youtube_nlp.research.settings import Study
from youtube_nlp.config import Config


def approved(identifier='v'):
    return {'id': identifier, 'title': 'Lesson', 'channel_id': 'ch', 'coder1':'Alice', 'coder2':'Bob',
            'decision1':'include', 'decision2':'include', 'format':'lecture'}


def test_independent_screening_and_consensus(tmp_path):
    row = approved()
    assert decision(row) == 'include'
    row['decision2'] = 'exclude'
    assert decision(row) == 'pending'
    row['consensus'] = 'include'
    assert decision(row) == 'include'
    row['coder2'] = 'Alice'
    assert decision(row) == 'pending'


def test_discovery_template_preserves_researcher_decisions(tmp_path):
    path=tmp_path/'screen.csv'
    table(path, [approved()])
    update_template(path, [{'id':'v','title':'changed'}, {'id':'new','title':'new'}])
    from youtube_nlp.research.storage import csv_rows
    rows=csv_rows(path)
    assert rows[0]['decision1'] == 'include'
    assert rows[0]['title'] == 'Lesson'
    assert report(path)['pending'] == 1


def test_replies_paginate_and_preserve_parent_and_creator(tmp_path):
    client=Mock()
    video={'id':'v', 'snippet':{'channelId':'ch'}}
    def get(resource, **params):
        if resource=='commentThreads':
            return {'items':[{'snippet':{'topLevelComment':{'id':'top','snippet':{'textOriginal':'question'}},'totalReplyCount':2}}]}
        if not params.get('pageToken'):
            return {'items':[{'id':'r1','snippet':{'textOriginal':'reply','authorChannelId':{'value':'ch'}}}], 'nextPageToken':'next'}
        return {'items':[{'id':'r2','snippet':{'textOriginal':'reply2'}}]}
    client._get.side_effect=get
    result=collect_comments(client, video, tmp_path/'checkpoint.json')
    assert result['complete']
    assert len(result['comments'])==3
    assert result['comments'][1]['parent_id']=='top'
    assert result['comments'][1]['is_creator'] is True
    assert 'authorChannelId' not in result['comments'][1]
    assert result['issues']==[]


def test_repeated_tokens_cannot_loop_indefinitely():
    client=Mock()
    client._get.return_value={'items':[], 'nextPageToken':'same'}
    with pytest.raises(ValueError, match='Repeated page token'):
        list(pages(client,'search',q='lesson'))


def test_unscreened_collection_is_blocked(tmp_path):
    study=Study(Config('', 'test', 1, 1, 'en', 2, 3, 42, tmp_path), tmp_path, {})
    with pytest.raises(ValueError,match='No eligible'):
        collect(study,Mock())


def test_year_windows_registry_parameters_and_template_creation(tmp_path):
    study=Study(Config('', 'test', 1, 1, 'en', 2, 3, 42, tmp_path), tmp_path,
                {'queries':[{'id':'q','q':'VTuber lesson','regions':['US'],'orders':['date']}], 'year_windows':[2020], 'max_search_pages_per_query':1})
    client=Mock()
    client._get.return_value={'items':[{'id':{'videoId':'v'},'snippet':{'channelId':'ch','title':'Lesson'}}]}
    discover(study,client)
    kwargs=client._get.call_args.kwargs
    assert kwargs['publishedAfter']=='2020-01-01T00:00:00Z'
    assert kwargs['publishedBefore']=='2021-01-01T00:00:00Z'
    assert (tmp_path/'screening.csv').exists()
    assert (tmp_path/'channel_screening.csv').exists()


def test_registry_redacts_credentials_and_author_profiles(tmp_path, monkeypatch):
    import json
    from youtube_nlp.research.collection import RegisteredClient
    from youtube_nlp.youtube import YouTubeClient
    payload={'items':[{'id':'c','snippet':{'authorDisplayName':'Private Name','authorChannelId':{'value':'owner'},'textOriginal':'Hello'}}]}
    monkeypatch.setattr(YouTubeClient,'_get',lambda *a,**k:payload)
    client=RegisteredClient('secret-api-key',tmp_path)
    client.context={'query_id':'Q1'}
    result=client._get('comments',parentId='parent')
    assert result['items'][0]['snippet']['authorChannelId']['value']=='owner'
    snapshot=next((tmp_path/'api_responses').glob('*.json')).read_text()
    assert 'authorDisplayName' not in snapshot
    registry=(tmp_path/'query_registry.jsonl').read_text()
    assert 'secret-api-key' not in registry
    assert json.loads(registry)['context']['query_id']=='Q1'
    client.close()


def test_comment_disabled_is_reported_not_silently_dropped(tmp_path):
    from youtube_nlp.youtube import YouTubeError
    client=Mock()
    client._get.side_effect=YouTubeError(403,'commentsDisabled')
    result=collect_comments(client,{'id':'v','snippet':{'channelId':'ch'}},tmp_path/'c.json')
    assert result['complete']
    assert result['issues'][0]['reason']=='commentsDisabled'


def test_inclusion_requires_observed_teaching_evidence():
    from youtube_nlp.research.screening import require_teaching_evidence
    row=approved()
    with pytest.raises(ValueError,match='avatar_evidence'):
        require_teaching_evidence(row)
    row['avatar_evidence']='Avatar visibly presents the explanation.'
    with pytest.raises(ValueError,match='instruction_evidence'):
        require_teaching_evidence(row)
    row['instruction_evidence']='Explains a concept and demonstrates a worked example.'
    row['instruction_timestamps']='02:15–06:40'
    require_teaching_evidence(row)


def test_default_video_queries_keep_identity_and_instruction_together():
    import json
    config=json.loads((Path(__file__).resolve().parents[2]/'research/study.json').read_text())
    for query in config['queries']:
        if query['type']=='video':
            assert '|' not in query['q']
            assert any(word in query['q'].lower() for word in ['vtuber','virtual youtuber','virtual teacher'])
            assert any(word in query['q'].lower() for word in ['lesson','tutorial','teaching','explanation','lecture'])
