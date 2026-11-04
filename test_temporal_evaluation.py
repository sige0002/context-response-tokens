import pytest
from evaluate_temporal import classification_metrics


def test_macro_recall_handles_missing_and_imbalanced_classes():
    result = classification_metrics([0, 0, 0, 1], [0, 0, 0, 0], 3)
    assert result["accuracy"] == 0.75
    assert result["macro_recall_present_classes"] == 0.5
    assert result["classes_present"] == 2
    assert result["support"] == [3, 1, 0]
    assert result["confusion_matrix"] == [[3, 0, 0], [1, 0, 0], [0, 0, 0]]
    with pytest.raises(ValueError):
        classification_metrics([0], [], 3)
