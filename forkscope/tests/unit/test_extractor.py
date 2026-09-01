from forkscope.extractor.mcq import MCQExtractor


def test_mcq_patterns():
    ex = MCQExtractor()
    assert ex.extract("So the answer is (B).") == "B"
    assert ex.extract("The correct answer is: C") == "C"
    assert ex.extract("ANSWER IS a") == "A"
    assert ex.extract("first I thought A, but finally the answer is (D)") == "D"
    assert ex.extract("hmm, let me think...") == "Other"
    assert ex.extract("option (A)") == "A"


def test_last_match_wins():
    ex = MCQExtractor()
    text = "The answer is (A). Wait, no. The answer is (C)."
    assert ex.extract(text) == "C"
