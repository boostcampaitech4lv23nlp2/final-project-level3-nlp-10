clean: clean-pyc clean-test
quality: set-style-dep check-quality
style: set-style-dep set-style
setup: set-precommit set-style-dep set-test-dep set-git set-dev set-mecab
test: set-test-dep set-test


##### basic #####
set-git:
	git config --local commit.template .gitmessage

set-style-dep:
	pip3 install isort==5.10.1 black==22.10.0 flake8==5.0.4

set-test-dep:
	pip3 install pytest==7.0.1

set-precommit:
	pip3 install pre-commit==2.17.0
	pre-commit install

set-dev:
	pip3 install -r requirements.txt

set-test:
	python3 -m pytest tests/

set-style:
	black --config pyproject.toml .
	isort --settings-path pyproject.toml .
	flake8 .

set-mecab:
	apt-get update
	apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl git
	python3 -m pip install --upgrade pip
	python3 -m pip install konlpy
	bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

check-quality:
	black --config pyproject.toml --check .
	isort --settings-path pyproject.toml --check-only .
	flake8 .

#####  clean  #####
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache
