# SimGuardAgent에 에이전트 연결하기: shadow 환경

SimGuardAgent는 타겟 에이전트의 모델 출력(Plan)을 실행 전에 가로채서, Simulation Manager가 한 경로로 보내고, 시뮬레이션한 결과를 Judge가 판정해 ALLOW 또는 BLOCK을 정한다.

도구 호출을 시뮬레이션할 때 **도구 결과를 어디서 얻는지**가 정확도를 정한다.

| 단계 | 사용자가 할 일 | 도구 결과 |
|---|---|---|
| 1. 기본값 | `protect()` 한 줄 | 내부 모델이 예상 |
| 2. 제공되는 shadow | 설정만 넘김 | 복사본·트랜잭션·테스트 인스턴스에서 실제로 실행한 결과 |
| 3. 직접 만든 shadow | `shadow_execute` 하나 구현 | 사용자가 정한 테스트 환경에서 실행한 결과 |

어느 단계든 판정은 항상 한다. shadow가 없는 도구는 내부 모델이 결과를 예상해서 판정하므로, 설정을 빠뜨려도 판정 없이 통과하지 않는다. 정확도만 낮아진다.

---

## 1단계: 설정 없이

```python
import simguard
simguard.protect()          # 에이전트 프레임워크를 import하기 전에 부른다
```

- OpenAI SDK와 LiteLLM으로 나가는 타겟 모델 호출을 가로챈다.
- 시뮬레이터와 Judge는 타겟 모델이 받은 입력 전체(시스템 프롬프트, 대화, 도구 호출과 결과, 도구 설명)를 본다.
- shadow가 없으므로 도구 결과는 내부 모델이 예상한다. 앞서 실행된 도구의 실제 결과는 대화에 있으므로 예상의 근거로 쓴다.

---

## 2단계: 제공되는 shadow

`protect(shadow_executors={도구 이름: 실행기})`로 넘긴다. 여기 있는 도구는 예상 대신 shadow에서 실제로 실행한 결과로 판정한다.

### 도구 함수: GET 전용 실제 실행 (`LiveReadExecutor`, 기본으로 켜짐)

에이전트와 같은 프로세스에 실제 도구가 있으면(LangChain, smolagents, `register_tool`), 도구를 **GET 전용 모드**로 실제로 실행한다. 어떤 요청을 보낼 도구인지 미리 분류하지 않고, 나가는 요청을 하나씩 검사한다.

| 실행 중 나온 동작 | 처리 |
|---|---|
| GET·HEAD·OPTIONS HTTP 요청 (80·443 포트) | 실제로 보낸다. 실제 응답으로 판정한다. |
| POST·PUT·PATCH·DELETE | 보내기 전에 멈춘다 → 내부 모델이 예상 |
| 파일 쓰기·삭제, 프로세스 실행, 다른 포트 연결(DB, SMTP), SQLite 연결 | 멈춘다 → 내부 모델이 예상 |
| `live_read_hosts` 밖의 도메인 (지정한 경우) | 멈춘다 → 내부 모델이 예상 |

```python
simguard.protect()                                                # 기본: 켜짐, 모든 도메인
simguard.protect(live_read_hosts={"api.weather.com", "github.com"})  # 이 도메인(하위 포함)에만 GET
simguard.protect(live_reads=False)                                # 끄기: 항상 예상
```

- `requests`, `urllib`, `urllib3`, `httpx`, `aiohttp` 요청과 파이썬 audit hook으로 보이는 부작용을 검사한다. 검사는 SimGuardAgent가 판정용으로 실행하는 호출에만 걸리고, 도구가 만든 스레드에도 이어진다. 에이전트의 실제 실행에는 걸리지 않는다.
- 멈추기 전에 보낸 GET은 이미 나간 상태다. 판정이 BLOCK이어도 그 GET은 되돌릴 수 없다. GET 쿼리에 담긴 데이터도 그 서버로 간다. 이것이 걱정되면 `live_read_hosts`로 도메인을 제한한다.
- 보지 못하는 것: C 확장이 직접 하는 네트워크·파일 작업(psycopg 같은 DB 드라이버, gRPC), 서버 쪽에서 상태를 바꾸는 GET. MCP 도구는 서버 프로세스가 요청을 보내므로 실행하지 않고 예상한다. SQL 호출은 아래 DB 경로로 간다.

### DB: 넘기지 않으면 내부 DB (`InternalDatabase`)

DB를 넘기지 않아도 인자에 SQL 문이 있는 도구 호출은 SimGuardAgent 내부 DB에서 실제로 실행해 본다. 판정할 때마다 SQLite 메모리 DB를 새로 만들고 버린다. 사용자의 DB에는 접근하지 않는다.

- 재구성 근거: 에이전트 입력에 있는 `CREATE TABLE` 문(시스템 프롬프트, 도구 설명), 앞서 실행된 조회의 실제 결과 행, 그래도 없는 테이블·컬럼은 이번 SQL이 참조하는 대로 만들고 자리표시 행을 넣는다.
- 알 수 있는 것: SQL이 실제로 하는 일(테이블 삭제, 조건 없는 삭제·수정, 읽는 컬럼), 문법 오류.
- 알 수 없는 것: 실제 데이터 값과 행 수. 결과에 이 사실을 적어 Judge에 넘긴다.
- 다른 DB 문법으로 쓴 SQL은 `sqlglot`이 설치돼 있으면 SQLite 문법으로 바꿔 다시 실행한다 (선택).

### DB: `protect(database=...)`

SimGuardAgent는 에이전트가 어떤 DB를 쓰는지 알 수 없다. SimGuardAgent를 연결하는 곳에 **에이전트가 쓰는 DB**를 넣으면, 인자에 SQL 문이 있는 도구 호출을 내부 DB 대신 이 DB에서 실행한 결과로 판정한다. 보통은 파일 경로나 DB URL 문자열 하나면 된다. 연결 코드(`lambda`)는 URL로 표현할 수 없는 드라이버나 인증 방식일 때 쓴다.

```python
import simguard

simguard.protect(database=lambda: psycopg.connect(os.environ["DATABASE_URL"]))   # 연결 코드 (어떤 드라이버든)
# simguard.protect(database="app.db")                                            # SQLite 파일
# simguard.protect(database="postgresql://user@host/db")                         # SQLAlchemy URL 또는 Engine

# DB가 여럿이면 도구별로
simguard.protect(databases={"query_orders": orders_connect, "query_users": users_connect})
```

- SQLite: 판정할 때마다 그 순간의 실제 DB를 임시 파일로 복사하고(`sqlite3` backup) 복사본에서 실행한다. 경로로 받으면 실제 DB 파일은 읽기 전용으로만 연다.
- 그 외 DB: 실제 DB의 트랜잭션 안에서 실행하고 되돌린다 (아래 `DatabaseShadowExecutor`와 같다). 복사본에서 실행하려면 복사본 DB(스테이징, DB 브랜치)에 연결하는 코드를 넘긴다.
- 연결 코드는 SimGuardAgent가 판정할 때마다 불러 새 연결을 만들고, 판정이 끝나면 닫는다. 에이전트의 연결은 건드리지 않는다.

### DB: `DatabaseShadowExecutor`

도구 호출을 실제 DB의 트랜잭션 안에서 실행하고, 결과와 상태 변화(바뀐 행 수, 스키마 변화)를 본 뒤 되돌린다. 실제 DB에는 남지 않는다. `copy=True`면 SQLite DB를 복사해 복사본에서 실행한다.

```python
from simguard import DatabaseShadowExecutor, protect

protect(shadow_executors={
    "execute_sql": DatabaseShadowExecutor("app.db"),                            # SQLite 파일
    # DatabaseShadowExecutor("postgresql://user@host/db")                       # SQLAlchemy URL
    # DatabaseShadowExecutor(engine)                                            # SQLAlchemy Engine
    # DatabaseShadowExecutor(lambda: psycopg.connect(DSN))                      # DB-API 연결 함수
})
```

- 인자에 SQL 문(`query`, `sql`, `statement` 등)이 있으면 그대로 실행한다.
- 도구가 SQL을 직접 받지 않으면, 그 도구가 하는 DB 작업을 `run`으로 넘긴다.

  ```python
  def reset_password(conn, tool_name, args):
      return conn.execute("UPDATE users SET password=? WHERE id=?", ("tmp", args["user_id"])).rowcount

  DatabaseShadowExecutor("app.db", run=reset_password)
  ```
- **되돌릴 수 없는 문장은 실행하지 않는다.** MySQL·MariaDB·Oracle은 `CREATE`, `ALTER`, `DROP`, `TRUNCATE`, `RENAME`을 실행하면 트랜잭션과 상관없이 바로 확정한다. 이런 문장은 shadow에서 실행하지 않고, 내부 모델이 결과를 예상해 판정한다 (shadow가 없을 때와 같다). 직접 만든 shadow도 안전하게 실행할 수 없는 호출이면 `None`을 돌려주면 된다.
- 여러 문장을 한 번에 보내면 실행하지 않고 오류로 돌려준다(SQLite).

### MCP: `MCPShadowExecutor`

MCP 도구 호출을 **테스트용 MCP 서버 인스턴스**에서 실행한다. 에이전트가 쓰는 실제 MCP 서버와는 따로 연결한다.

```python
from mcp import StdioServerParameters
from simguard import MCPShadowExecutor, protect

# 같은 서버 프로그램을 테스트 설정으로 따로 띄운다
test_server = MCPShadowExecutor(StdioServerParameters(
    command="python", args=["notes_server.py"],
    env={"NOTES_DB": "test_notes.db", "MAIL_API_KEY": "sandbox-key"},
))
# 또는 이미 떠 있는 테스트·스테이징 서버: MCPShadowExecutor("https://staging.example.com/mcp")

protect(shadow_executors=dict.fromkeys(["add_note", "delete_note", "send_mail"], test_server))
```

- **shadow의 안전성은 테스트 인스턴스 설정에 달려 있다.** stdio 서버는 호출마다 새로 띄우므로 서버 프로세스 안의 상태는 남지 않는다. 그러나 서버가 외부 서비스(메일, 결제, 실제 DB)에 연결하면 그 호출은 실제로 일어난다. 테스트 인스턴스는 반드시 테스트용 외부 자원에 연결되게 설정한다. 실제 서버 설정을 그대로 넘기면 shadow가 아니다.
- 상태 변화를 보려면 `snapshot`으로 서버의 읽기 도구를 부르는 함수를 넘긴다.

  ```python
  async def notes(session):
      return (await session.call_tool("list_notes", {})).content[0].text

  MCPShadowExecutor(params, snapshot=notes)
  ```
- 테스트 서버에 연결하지 못하면 그 사실을 오류로 넘기고, Judge가 시도한 호출을 보고 판정한다.

### 코드·셸: `ExecutionEnvironment`

코드와 셸 명령은 ExecutionSimulator가 Docker 컨테이너에서 작업 폴더 복사본으로 실행한다.

```python
from simguard import ExecutionEnvironment, protect
protect(execution_environment=ExecutionEnvironment(workdir="/home/me/project"))
```

### 메모리 안의 환경 객체: `RuntimeShadowToolExecutor`

도구가 파이썬 환경 객체의 상태를 바꾸는 경우(AgentDojo 등), 환경을 deep copy해서 실행하고 전후 상태를 비교한다.

---

## 3단계: 직접 만든 shadow

SimGuardAgent가 알 수 없는 서비스(결제, 사내 API)는 `shadow_execute` 하나만 구현한다. async 함수여도 된다.

```python
from simguard import ShadowExecutionResult, protect

class PaymentShadow:
    def shadow_execute(self, tool_name, tool_arguments):
        # 예: 결제 서비스의 테스트 모드 키로 같은 요청을 보낸다
        response = payments_test_client.transfer(**tool_arguments)
        return ShadowExecutionResult(result=str(response), error=None,
                                     state_diff=f"test transfer {response.id} created")

protect(shadow_executors={"send_money": PaymentShadow()})
```

많은 외부 서비스가 테스트 환경(결제 test mode, 메일 sandbox, 스테이징 API)을 제공한다. 그런 경우 테스트 주소나 테스트 키로 보내도록 연결하면 된다.
