run:
    python advisor.py

init:
	pip install -r requirements.txt

clean:
    rm -rf __pycache__

venv/bin/activate: requirements.txt
 	python3 -m venv venv
 	./venv/bin/pip install -r requirements.txt

freeze:
	pip freeze > requirements.txt