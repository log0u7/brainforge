def test_prompt_formats_with_material():
    from brainforge.experiments.model_only_pilot import MODEL_ONLY_PROMPT
    out = MODEL_ONLY_PROMPT.format(material="int main(){ return 0; }")
    assert "int main(){ return 0; }" in out
    assert '"vuln_type"' in out  # the JSON example survives
