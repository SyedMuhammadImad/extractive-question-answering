import pytest
from qa import exact_match, token_f1, answer, evaluate

def test_normalization():
    assert exact_match('The cat!','cat')==1
    assert exact_match(' New   York ','new york')==1

def test_multiset_f1():
    assert token_f1('cat cat cat dog','cat dog')==pytest.approx(2/3)
    assert token_f1('','')==1
    assert token_f1('cat','')==0

def test_invalid_input_and_multi_reference():
    with pytest.raises(ValueError):answer(None,'','who?')
    report=evaluate(lambda **kw:{'answer':'cat'},[{'id':'example','context':'cat','question':'what?','answers':['dog','the cat']}])
    assert report['exact_match']==1 and report['f1']==1
    with pytest.raises(ValueError):evaluate(None,[])
