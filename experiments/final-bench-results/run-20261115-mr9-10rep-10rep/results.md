# Bench lab results

repetitions: 10

| model | successes/attempts | success_rate |
|---|---|---|
| base | 168/290 | 0.58 |
| trained | 169/290 | 0.58 |

attempts without LLM facts: base 9/290, trained 37/290

contamination: LEAKED: ['ctf-lictor-callme-002-001', 'ctf-lictor-fmtstr-001', 'ctf-lictor-fmtstr-002-001', 'ctf-lictor-orw-001', 'ctf-lictor-orw-002-001', 'ctf-lictor-ret2csu-001', 'ctf-lictor-ret2func-001', 'ctf-lictor-ret2func-002-001', 'ctf-lictor-ret2libc-001-001', 'ctf-lictor-ret2libc-002-001', 'ctf-lictor-ret2plt-001', 'ctf-lictor-ret2plt-002-001', 'ctf-lictor-shellcode-001', 'ctf-lictor-shellcode-002-001', 'ctf-lictor-srop-001-001', 'ctf-lictor-uaf_heap-002-001', 'ctf-lictor-uaf_train_a-001', 'ctf-lictor-uaf_train_b-001']
