outline with info here: https://docs.google.com/document/d/10P-uX-Xp8bp0BnvZsNPiZsg1TYci1FXrIZWcyKgpq0o/edit?usp=sharing

result plots in data/processed/runs/pilot3/results/plots

questions rn:
- the random paragraphs text for random replacement corruption is currently fully in english (random sentences from wikipedia, found a dataset in kaggle), would that be potentially problematic/should i look for datasets in each language instead?
- should i add any charts/graphs?
- for final run before ACSEF settings, should i change anything here: (1) Small LLMs: Llama 3.1 8B Instruct, Qwen2.5 7B Instruct, Gemma 2 9B it, (2) Judge LLMs: Gemini 2.5 pro, Deepseek R1, (3) languages: english, german, greek, romanian, (4) 100 claims per language, (5) 7 articles retrieved via SerpAPI per claim, (6) minimum & maximum characters per article are 800 and 10000 respectively, (7) timeout for webpage fetch requests in seconds is 25s (is this needed), (8) BM25 top 5 articles per claim are chosen, (9) corruption at 30%, 50%, 70%, (10) should i have limits on the maximum tokens a model can output?
