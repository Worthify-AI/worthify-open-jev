from types import SimpleNamespace
from openjev_phase1.direct import _forward


class Logits:
    def __getitem__(self, key):
        return 'last-position-logits'


def test_peft_style_wrapper_uses_base_logit_limit():
    class Base:
        def forward(self, input_ids=None, logits_to_keep=None):
            pass

    class Adapter:
        def get_base_model(self):
            return Base()

        def forward(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(logits=Logits())

        __call__ = forward

    adapter = Adapter()
    assert _forward(adapter, {'input_ids': [1, 2, 3]}) == 'last-position-logits'
    assert adapter.kwargs['logits_to_keep'] == 1
    assert adapter.kwargs['use_cache'] is False


def test_cli_warmup_is_discarded_and_all_rows_are_measured(tmp_path, monkeypatch):
    import json
    import sys
    from openjev_phase1 import cli

    rows = [{'id': str(i), 'state': 'Evidence', 'question': 'Decide',
             'options': [{'id': 'yes', 'description': 'Yes'}, {'id': 'no', 'description': 'No'}]}
            for i in range(2)]
    source, output = tmp_path/'input.jsonl', tmp_path/'output.jsonl'
    source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    calls = []
    monkeypatch.setattr(cli, 'load_causal_model', lambda *a, **kw: (None, None, {}))

    def scorer(model, tokenizer, row, metadata, max_tokens):
        calls.append(row['id'])
        return {'id': row['id'], 'option_ids': ['yes', 'no'], 'probabilities': [.6, .4]}

    monkeypatch.setattr(cli, 'direct_score', scorer)
    monkeypatch.setattr(sys, 'argv', ['score', '--mode', 'direct', '--model', 'model', '--revision', 'r',
                                    '--input', str(source), '--output', str(output), '--warmup', '3'])
    cli.main()
    predictions = [json.loads(line) for line in output.read_text().splitlines()]
    assert calls == ['0', '1', '0', '0', '1']
    assert [row['id'] for row in predictions] == ['0', '1']
    assert all(row['warm'] for row in predictions)
