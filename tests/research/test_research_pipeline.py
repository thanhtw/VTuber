from pathlib import Path
import json
import numpy as np
import pytest
from youtube_nlp.config import Config
from youtube_nlp.research.settings import Study
from youtube_nlp.research.storage import save
from youtube_nlp.research.pipeline import run_analysis

RULES = Path(__file__).resolve().parents[2] / 'research/aspect_rules.json'


class FakeScorer:
    identifier='synthetic-test-only'
    revision='fixture'
    backend='test'
    def score(self,texts):
        return [{'polarity':'positive','positive':.8,'neutral':.1,'negative':.1,'compound':.7} for _ in texts]


def translator(rows,config):
    return [{**row,'text_en':row['text'],'source_language':'en'} for row in rows]


def topic_fitter(videos,study,output):
    output.mkdir(parents=True)
    return [{'topic_id':1,'name':'Synthetic topic','family':'Test'}], {v['video_id']:[1.] for v in videos}


def fixture_study(tmp_path):
    options={'aspect_rules_path':str(RULES),'bootstrap_samples':10,'permutations':10,
             'fixed_margin_permutations':3,'community_null_samples':3,'community_bootstrap_samples':3,
             'audit_sample_per_language':2,'seed':42}
    api=Config('', 'test', 1, 1, 'en', 2, 3, 42, tmp_path)
    study=Study(api,tmp_path,options)
    channels=[{'id':'ch1','snippet':{'publishedAt':'2020-01-01T00:00:00Z'},'statistics':{'hiddenSubscriberCount':True,'videoCount':'1'}},
              {'id':'ch2','snippet':{'publishedAt':'2020-01-01T00:00:00Z'},'statistics':{'subscriberCount':'100','videoCount':'1'}}]
    videos=[{'id':f'v{i}','format':'lecture','teaching_eligibility_evidence':{'avatar_evidence':'Synthetic test presenter','instruction_evidence':'Synthetic test lesson','instruction_timestamps':'00:00–01:00'},'snippet':{'channelId':f'ch{i}','title':'Clear lesson','description':'A lesson','publishedAt':'2020-01-01T00:00:00Z'},'statistics':{'viewCount':'100'},'contentDetails':{}} for i in (1,2)]
    comments=[{'comment_id':f'c{i}','video_id':f'v{1 if i < 4 else 2}','channel_id':f'ch{1 if i < 4 else 2}',
               'parent_id':None,'text':text,'is_reply':False,'is_creator':False} for i,text in enumerate([
                   'Clear explanation. I understand.','Cute avatar keeps me company.','Hello everybody!','Too fast. Clear explanation.',
                   'I do not understand.','Clear voice. Clear explanation.','Study together. Cute avatar.',''])]
    save(tmp_path/'dataset.json',{'collection_complete':True,'collected_at':'2026-10-17T00:00:00Z',
                                 'channels':channels,'videos':videos,'comments':comments,'coverage_issues':[]})
    return study


def test_research_pipeline_exports_auditable_rq_outputs(tmp_path):
    study=fixture_study(tmp_path)
    output=run_analysis(study,scorer=FakeScorer(),topic_fitter=topic_fitter,translator=translator)
    assert (output/'rq3/heterogeneous.graphml').exists()
    assert (output/'rq3/aspect_network.png').exists()
    data=json.loads((output/'analyzed_dataset.json').read_text())
    assert all('text_en' in c and 'text' in c for c in data['comments'])
    assert all(c['comment_id'].startswith('comment_') for c in data['comments'])
    assert len(data['comments'])==8
    assert sum(c['usable_text'] for c in data['comments'])==7
    assert any(not c['aspects'] for c in data['comments'])
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['sentiment_model']=='synthetic-test-only'
    assert 'provisional' in manifest['status']
    assert (tmp_path/'private_identifier_map.json').exists()
    assert json.loads((tmp_path/'latest_analysis.json').read_text())['path']==str(output)
    # A second run creates a different directory instead of overwriting empirical outputs.
    second=run_analysis(study,scorer=FakeScorer(),topic_fitter=topic_fitter,translator=translator)
    assert output != second


def test_incomplete_research_collection_is_not_analyzed(tmp_path):
    study=fixture_study(tmp_path)
    data=json.loads((tmp_path/'dataset.json').read_text())
    data['collection_complete']=False
    save(tmp_path/'dataset.json',data)
    with pytest.raises(ValueError,match='incomplete'):
        run_analysis(study,scorer=FakeScorer(),topic_fitter=topic_fitter,translator=translator)
