.PHONY: analyze app benchmark deploy-selected lint physics test

app:
	streamlit run app.py

test:
	python -m unittest discover -s tests -v
	python -m experiments.scan_public_language

lint:
	ruff check .

physics:
	python physics_cli.py

benchmark:
	python -m experiments.run_retrieval_benchmark

deploy-selected:
	python -m experiments.deploy_selected_index

analyze:
	python -m experiments.analyze_results
	python -m experiments.build_notebook
	python -m experiments.qa_results
