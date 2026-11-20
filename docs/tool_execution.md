# SimGuardAgent에 에이전트 연결하기: 도구 결과를 얻는 방법

SimGuardAgent는 타겟 에이전트의 모델 출력(Plan)을 실행 전에 가로채서, Simulation Manager가 한 경로로 보내고, 시뮬레이션한 결과를 Judge가 판정해 ALLOW 또는 BLOCK을 정한다.

도구 호출을 시뮬레이션할 때 **도구 결과를 어디서 얻는지**가 정확도를 정한다.

| 호출 | 도구 결과 | 사용자가 할 일 |
|---|---|---|
| 실제 도구가 있는 호출 | 도구를 GET 전용 모드로 실제 실행한 결과. GET이 아닌 요청이나 파일 쓰기가 나오면 멈추고 내부 모델이 예상 | 자동으로 찾지 못하는 도구는 `protect(tools=...)`로 알려 준다 |
| SQL 인자가 있는 호출 | 에이전트 DB의 복사본·롤백 트랜잭션에서 실행한 결과. DB를 알려 주지 않으면 SimGuardAgent 내부 SQLite에서 실행 | `protect(database=...)` (선택) |
| 코드·셸 | Docker 컨테이너(네트워크 차단)에서 행동마다 새로 실행한 결과 | 없음 (Docker 필요) |
| 그 외 (MCP 도구, 실행을 멈춘 호출 등) | 내부 모델이 앞 단계의 실제 도구 결과와 인자를 보고 예상 | 없음 |

어느 경우든 판정은 항상 한다. 실제로 실행할 수 없는 호출은 내부 모델이 결과를 예상해서 판정하므로, 설정을 빠뜨려도 판정 없이 통과하지 않는다. 정확도만 낮아진다.

---

## 설정 없이

```python
import simguard
simguard.protect()          # 에이전트 프레임워크를 import하기 전에 부른다
```

- OpenAI SDK와 LiteLLM으로 나가는 타겟 모델 호출을 가로챈다.
- 시뮬레이터와 Judge는 타겟 모델이 받은 입력 전체(시스템 프롬프트, 대화, 도구 호출과 결과, 도구 설명)를 본다.
- 앞서 실행된 도구의 실제 결과는 대화에 있으므로 예상의 근거로 쓴다.

---

## 실제 도구: GET 전용 실행 (`LiveReadExecutor`, 기본으로 켜짐)

에이전트의 실제 도구를 판정 전에 **GET 전용 모드**로 실행한다. 어떤 요청을 보낼 도구인지 미리 분류하지 않고, 나가는 요청을 하나씩 검사한다.

| 실행 중 나온 동작 | 처리 |
|---|---|
| GET·HEAD·OPTIONS HTTP 요청 (80·443 포트) | 실제로 보낸다. 실제 응답으로 판정한다. |
| POST·PUT·PATCH·DELETE | 보내기 전에 멈춘다 → 내부 모델이 예상 |
| 파일 쓰기·삭제, 프로세스 실행, 다른 포트 연결(DB, SMTP), SQLite 연결 | 멈춘다 → 내부 모델이 예상 |
| `live_read_hosts` 밖의 도메인 (지정한 경우) | 멈춘다 → 내부 모델이 예상 |

```python
simguard.protect()                                                   # 기본: 켜짐, 모든 도메인
simguard.protect(live_read_hosts={"api.weather.com", "github.com"})  # 이 도메인(하위 포함)에만 GET
simguard.protect(live_reads=False)                                   # 끄기: 항상 예상
```

### 실제 도구 알려 주기: `protect(tools=...)`

에이전트의 실제 도구는 연결하는 곳에서 알려 준다. 어떤 프레임워크의 도구가 들어올지 모르므로 SimGuardAgent는 특정 프레임워크를 가정하지 않고, 도구 객체를 공통 규칙으로 읽는다.

```python
simguard.protect(tools=[send_email(), search_hotels()])     # 목록: 이름은 도구에서 읽는다
simguard.protect(tools={"send_email": my_send_email})        # {이름: 도구}
simguard.register_tools([...])                               # protect 밖에서 등록
```

- 읽는 규칙 (`simguard.as_real_tool`):
  - 이름: `name` 속성 > 함수 이름. 이름을 읽을 수 없으면 `{이름: 도구}`로 넘긴다.
  - 설명: `description` 속성 > docstring.
  - 인자: `parameters` / `input_schema` / `tool_call_schema` / `args_schema` 속성(JSON 스키마 또는 `model_json_schema()`를 가진 클래스) > `inputs` 속성 > 함수 시그니처.
  - 실행: `invoke(인자)` 메서드 > `도구(**인자)`. 동기·async 모두.
- 같은 이름의 서로 다른 도구가 여럿 등록되면 어느 것인지 알 수 없으므로 실행하지 않고 예상한다.

### 주의

- 검사는 SimGuardAgent가 판정용으로 실행하는 호출에만 걸리고, 도구가 만든 스레드에도 이어진다. 에이전트의 실제 실행에는 걸리지 않는다.
- 멈추기 전에 보낸 GET은 이미 나간 상태다. 판정이 BLOCK이어도 그 GET은 되돌릴 수 없다. GET 쿼리에 담긴 데이터도 그 서버로 간다. 걱정되면 `live_read_hosts`로 도메인을 제한한다.
- 기준은 HTTP 동작이다. HTTP를 쓰지 않고 메모리만 바꾸는 도구(테스트용 가짜 도구 등)는 끝까지 실행된다.
- 보지 못하는 것: C 확장이 직접 하는 네트워크·파일 작업(psycopg 같은 DB 드라이버, gRPC), 서버 쪽에서 상태를 바꾸는 GET. MCP 도구는 서버 프로세스가 요청을 보내므로 실행하지 않고 예상한다. SQL 호출은 아래 DB 경로로 간다.

---

## DB

### 넘기지 않으면 내부 DB (`InternalDatabase`)

DB를 넘기지 않아도 인자에 SQL 문이 있는 도구 호출은 SimGuardAgent 내부 DB에서 실제로 실행해 본다. 판정할 때마다 SQLite 메모리 DB를 새로 만들고 버린다. 사용자의 DB에는 접근하지 않는다.

- 재구성 근거: 에이전트 입력에 있는 `CREATE TABLE` 문(시스템 프롬프트, 도구 설명), 앞서 실행된 조회의 실제 결과 행, 그래도 없는 테이블·컬럼은 이번 SQL이 참조하는 대로 만들고 자리표시 행을 넣는다.
- 알 수 있는 것: SQL이 실제로 하는 일(테이블 삭제, 조건 없는 삭제·수정, 읽는 컬럼), 문법 오류.
- 알 수 없는 것: 실제 데이터 값과 행 수. 결과에 이 사실을 적어 Judge에 넘긴다.
- 다른 DB 문법으로 쓴 SQL은 `sqlglot`이 설치돼 있으면 SQLite 문법으로 바꿔 다시 실행한다 (선택).

### 에이전트 DB: `protect(database=...)`

SimGuardAgent는 에이전트가 어떤 DB를 쓰는지 알 수 없다. 연결하는 곳에 **에이전트가 쓰는 DB**를 넣으면, 인자에 SQL 문이 있는 도구 호출을 내부 DB 대신 이 DB에서 실행한 결과로 판정한다.

```python
import simguard

simguard.protect(database=lambda: psycopg.connect(os.environ["DATABASE_URL"]))   # 연결 코드 (어떤 드라이버든)
# simguard.protect(database="app.db")                                            # SQLite 파일
# simguard.protect(database="postgresql://user@host/db")                         # SQLAlchemy URL 또는 Engine

# DB가 여럿이면 도구별로
simguard.protect(databases={"query_orders": orders_connect, "query_users": users_connect})
```

- SQLite: 판정할 때마다 그 순간의 실제 DB를 임시 파일로 복사하고(`sqlite3` backup) 복사본에서 실행한다. 경로로 받으면 실제 DB 파일은 읽기 전용으로만 연다.
- 그 외 DB: 실제 DB의 트랜잭션 안에서 실행하고 되돌린다.
- 연결 코드는 SimGuardAgent가 판정할 때마다 불러 새 연결을 만들고, 판정이 끝나면 닫는다. 에이전트의 연결은 건드리지 않는다.
- **되돌릴 수 없는 문장은 실행하지 않는다.** MySQL·MariaDB·Oracle은 `CREATE`, `ALTER`, `DROP`, `TRUNCATE`, `RENAME`을 실행하면 트랜잭션과 상관없이 바로 확정한다. 이런 문장은 실행하지 않고 내부 모델이 결과를 예상해 판정한다.
- 여러 문장을 한 번에 보내면 실행하지 않고 오류로 돌려준다(SQLite).

도구가 SQL을 직접 받지 않으면, 그 도구가 하는 DB 작업을 `DatabaseExecutor`의 `run`으로 넘긴다.

```python
from simguard import DatabaseExecutor, protect

def reset_password(conn, tool_name, args):
    return conn.execute("UPDATE users SET password=? WHERE id=?", ("tmp", args["user_id"])).rowcount

protect(databases={"reset_password": DatabaseExecutor("app.db", run=reset_password, copy=True)})
```

---

## 코드·셸

코드와 셸 명령은 ExecutionSimulator가 행동마다 새 Docker 컨테이너(네트워크 차단, 읽기 전용 루트, 메모리·CPU 제한)에서 실행한다. 실행할 수 없는 언어는 내부 모델이 실행 결과를 예측한다. Docker를 쓸 수 없으면 실행하지 않고 차단한다.
