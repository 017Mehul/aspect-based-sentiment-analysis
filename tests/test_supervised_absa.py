from pathlib import Path

from absa_supervised import evaluate, parse_semeval_xml, tokenize_with_offsets


def _write_xml(path: Path, sentences: list[tuple[str, str, str, str, str]]) -> None:
    # Tuples are (text, term, polarity, start, end).
    rows = ['<?xml version="1.0" encoding="UTF-8"?>', "<sentences>"]
    for i, (text, term, polarity, start, end) in enumerate(sentences):
        rows.append(
            f'<sentence id="{i}"><text>{text}</text><aspectTerms>'
            f'<aspectTerm term="{term}" polarity="{polarity}" from="{start}" to="{end}"/>'
            f"</aspectTerms></sentence>"
        )
    rows.append("</sentences>")
    path.write_text("\n".join(rows), encoding="utf-8")


def test_tokenizer_tracks_character_offsets():
    text = "Fresh coffee!"
    tokens = tokenize_with_offsets(text)
    assert tokens[1] == ("coffee", 6, 12)


def test_parser_rejects_misaligned_offsets(tmp_path):
    path = tmp_path / "bad.xml"
    _write_xml(path, [("The food is great.", "food", "positive", "5", "9")])
    examples = parse_semeval_xml(path)
    assert examples[0]["aspects"] == []


def test_supervised_absa_writes_separate_stage_metrics(tmp_path):
    train = tmp_path / "train.xml"
    test = tmp_path / "test.xml"
    _write_xml(train, [
        ("The food is good.", "food", "positive", "4", "8"),
        ("The service is bad.", "service", "negative", "4", "11"),
        ("Food was okay.", "Food", "neutral", "0", "4"),
        ("Service was awful.", "Service", "negative", "0", "7"),
        ("The meal is excellent.", "meal", "positive", "4", "8"),
        ("The staff was terrible.", "staff", "negative", "4", "9"),
    ])
    _write_xml(test, [
        ("The food tastes good.", "food", "positive", "4", "8"),
        ("The service feels bad.", "service", "negative", "4", "11"),
        ("Food seems okay.", "Food", "neutral", "0", "4"),
    ])

    result = evaluate(train, test, tmp_path / "outputs")

    assert result["train_sentences"] == 6
    assert result["test_sentences"] == 3
    assert "aspect_term_extraction_exact_span" in result
    assert "aspect_polarity_given_gold_aspects" in result
    assert "end_to_end_joint_exact_span_and_polarity" in result
    assert (tmp_path / "outputs" / "semeval_absa_metrics.json").is_file()
    assert (tmp_path / "outputs" / "supervised_absa_models.joblib").is_file()
