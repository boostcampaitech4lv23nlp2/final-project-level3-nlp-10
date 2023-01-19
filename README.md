# final-project-level3-nlp-10

## 📄 Guideline
> `make` 명령어를 실행했을 때, `bash: make: command not found` error가 발생한다면, `make`를 설치해주셔야 합니다.
>```bash
> apt-get update
> apt-get install -y make
>```

1. Setup

- precommit, style, pytest, gitmessage, requirements

```bash
make setup
```

2. Test
- implement pytest in `/tests/` path

```bash
make test
```

3. Execute code formatting & Check lint

```bash
make style
```
