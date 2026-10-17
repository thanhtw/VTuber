from youtube_nlp.research.review import validate_reviews, obvious_software_tutorial
from youtube_nlp.research.screening import update_template
from youtube_nlp.research.storage import table,csv_rows


def test_unsupported_quotes_cannot_advance_a_candidate():
    rows=validate_reviews([{'id':'x','kind':'video','source':'VIDEO TITLE: An interesting lesson'}], {'reviews':[{
        'id':'x','recommendation':'advance','priority':'high','evidence_quote':'An interesting lesson',
        'avatar_quote':'I am an avatar teacher','instruction_quote':'We teach mathematics',
        'rationale':'Candidate','exclusion_reason':'','suggested_format':'unknown'}]})
    assert rows[0]['recommendation']=='uncertain'
    assert not rows[0]['avatar_quote']


def test_vtuber_creation_is_excluded_but_subject_instruction_is_not():
    assert obvious_software_tutorial({'kind':'video','source':'VIDEO TITLE: How To Make A VTuber For FREE'})
    assert not obvious_software_tutorial({'kind':'video','source':'VIDEO TITLE: Accounting lesson with a VTuber'})
    assert not obvious_software_tutorial({'kind':'channel','source':'CHANNEL TITLE: I teach math and make a VTuber model'})


def test_template_preserves_ai_review_fields_and_human_fields(tmp_path):
    path=tmp_path/'screening.csv'
    table(path,[{'id':'x','title':'lesson','coder1':'Researcher','decision1':'include','ai_recommendation':'advance'}])
    update_template(path,[{'id':'x','title':'new title'}])
    row=csv_rows(path)[0]
    assert row['ai_recommendation']=='advance'
    assert row['coder1']=='Researcher'
    assert row['decision1']=='include'


def test_sparse_channel_metadata_does_not_justify_whole_channel_exclusion():
    source={'id':'ch','kind':'channel','source':'CHANNEL TITLE: Mixed channel\nCHANNEL DESCRIPTION: I play games.'}
    result=validate_reviews([source],{'reviews':[{'id':'ch','recommendation':'exclude','priority':'low',
        'evidence_quote':'I play games.','avatar_quote':'','instruction_quote':'',
        'rationale':'No educational metadata.','exclusion_reason':'Not educational','suggested_format':'unknown'}]})
    assert result[0]['recommendation']=='uncertain'
    assert result[0]['exclusion_reason']==''
