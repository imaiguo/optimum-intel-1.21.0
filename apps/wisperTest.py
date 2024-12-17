
- from transformers import AutoModelForSeq2SeqLM
+ from optimum.intel import OVModelForSeq2SeqLM
  from transformers import AutoTokenizer, pipeline

  model_id = "echarlaix/t5-small-openvino"
- model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
+ model = OVModelForSeq2SeqLM.from_pretrained(model_id)
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)
  results = pipe("He never went out without a book under his arm, and he often came back with two.")

  [{'translation_text': "Il n'est jamais sorti sans un livre sous son bras, et il est souvent revenu avec deux."}]
