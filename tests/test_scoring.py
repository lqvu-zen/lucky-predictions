from ml.score import hits_and_pos


def test_position_accuracy_example():
    # actual 1-2-3-4-5-6 vs guess 1-19-29-37-36-55 -> only the 1 is right,
    # and it's at the right (first) position, so 1 hit and 1 pos-hit
    hits, pos = hits_and_pos([1, 19, 29, 37, 36, 55], [1, 2, 3, 4, 5, 6], 6)
    assert hits == 1
    assert pos == 1


def test_right_number_wrong_position_counts_as_hit_not_pos():
    # 2 is present but not at its sorted position
    hits, pos = hits_and_pos([2, 10, 20, 30, 40, 50], [1, 2, 3, 4, 5, 6], 6)
    assert hits == 1
    assert pos == 0


def test_perfect_ticket():
    hits, pos = hits_and_pos([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1], 6)
    assert (hits, pos) == (6, 6)


def test_uses_only_main_numbers():
    # actual has a 7th bonus number that must be ignored
    hits, pos = hits_and_pos([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 99], 6)
    assert (hits, pos) == (6, 6)
