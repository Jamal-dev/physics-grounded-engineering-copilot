.PHONY: analyze app benchmark lint physics test

app:
	streamlit run app.py

test:
	python -m unittest discover -s tests -v

lint:
	ruff check .

physics:
	python physics_cli.py

benchmark:
	python -m experiments.run_retrieval_benchmark

analyze:
	python -m experiments.analyze_results
	python -m experiments.build_notebook
	python -m experiments.qa_results
