from finlang.nlp_utils import tagging_utils as tu


def test_tagging_pipeline(text_df):
    for task, settings in tu.TASK_SETTINGS.items():
        if task not in {'feature-extraction', 'sentiment-analysis', 'ner', 'summarization', 'translation_en_to_fr'}:
            continue
        proc = tu.tagging_pipeline(task)
        out = proc(text_df, source='text', progress=False)
        assert f"text{settings['suffix']}" in out.columns
        assert out[f"text{settings['suffix']}"].isna().sum() == 0
