# ★ 옛 로그로 셋을 셌다 — **최다 거부 사유가 우리 쪽 결함이다**

> **재현**: `python3 experiments/read_logs.py` (LLM 0회)
> **대상**: `F3rw-p8-s0..s5` (24라운드 6실행, 제안 1,728개)

## 1. 중복 — 5.9~8.7%, **exploit 에 몰린다**

| 실행 | 제안 | 중복 | 비율 | 부모 종류별 |
|---|---:|---:|---:|---|
| s0 | 288 | 24 | 8.3% | exploit 22 / explore 5 / cross 4 |
| s1 | 288 | 24 | 8.3% | exploit 20 / explore 3 / cross 1 |
| s2 | 288 | 24 | 8.3% | exploit 9 / explore 6 / cross 9 |
| s3 | 288 | 23 | 8.0% | exploit 14 / explore 5 / cross 4 |
| s4 | 288 | 17 | 5.9% | exploit 10 / explore 6 / cross 1 |
| s5 | 288 | 25 | 8.7% | exploit 14 / explore 6 / cross 5 |

```
★ 합계 exploit 89 · explore 31 · cross 24
   exploit 이 라운드당 6/12 인데 중복의 62% 를 낸다
같은 코드가 최대 5회까지 나온다 (실행마다 16~31회가 재등장)
```

⚠️ **어떤 가설이 배정됐을 때 중복이 나오나는 옛 로그로 못 센다** —
제안별 가설 배정이 **채택된 규칙에만** 남는다. 트레이스가 그 자리다.

## 2. 거부 — ★ 34건 중 34건이 **인프라 문제**

```
사유별 합계   run 34 · static 26 · sandbox 3
```

| 횟수 | 종류 | 사유 |
|---:|---|---|
| **34** | run | **`KeyError: '__import__'`** |
| 12 | static | AST 노드 401~465 > 400 |
| 1 | static | 등록되지 않은 형상 수준 값: `p.waves` |
| 1 | static | 허용되지 않은 numpy 함수: `np.exp2` |
| 3 | sandbox | 타입 오류 등 |

### ★ `KeyError: '__import__'` 는 **규칙 잘못이 아니다**

재현했다:

```python
code = "def score(f, p, hw, w):\n    s = np.log(f.waves - 5.0) * w[0]\n    return s"
fn = compile_rule(code); fn(F(), None, None, w)
-> ★ KeyError: '__import__'

np.seterr(all="ignore") 를 먼저 부르면
-> [nan nan nan]   (그리고 nan 은 우리가 명시적으로 처리한다)
```

**numpy 가 경고를 내려고 `warnings` 를 import 하는데, 제한된 builtins 에
`__import__` 가 없어서 죽는다.** 규칙이 `log(음수)` · 0 나누기 · overflow
를 한 번이라도 만들면 그 규칙은 **채점도 못 받고 버려진다.**

★ **샌드박스 자식에는 이미 이 방어가 있다** (`sandbox.py:137`,
`np.seterr(all="ignore")`, 주석까지 달려 있다). **채점·적합 경로에는
없다** — `loop.py` 의 `compile_rule` 자리 다섯 곳과 `canonical.py`.

```
버려진 제안   34개 / 1,728개 = 2.0%
그 규칙들이 나빴는지 좋았는지 ★ 우리는 모른다 (점수가 안 났다)
```

### AST 노드 상한은 **아슬아슬하게** 넘는다

12건 전부 401~465 다 (상한 400). 파라미터 8 의 `ast_nodes` 상한이고,
모델이 조금씩 넘긴다. **상한을 바꾸면 조건 변경이므로 여기서 안 고친다.**

### 거부가 뒤 라운드에 몰린다

```
r1~r7    1 2 4 4 3 1 2
r18~r23  4 4 5 8 3 9
```

규칙이 자라면서 노드 상한과 실행 오류가 함께 는다.

## 3. 죽은 항

(계산 중 — `read_logs.py` 의 셋째 절)

## ★ 고치지 않았다

지시문 §8: **"로그를 보고 바로 고치지 마라 — 고치면 조건 변경이다."**
`__import__` 결함은 고치면 버려지던 제안 2% 가 채점되므로 진화가 달라진다.
**무엇을 고칠지 정하고 실험 계획서를 먼저 쓴다.**
