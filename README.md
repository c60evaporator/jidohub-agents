# jidohub-agents

jidohub は自動運転向けの Agent / Dataset / Interface 共有プラットフォーム。
**jidohub-agents は Agent をロードして実行する Python API** であり、5 リポジトリ構成の**実行層**にあたる。

| リポジトリ | 役割 |
|---|---|
| jidohub-web | Agents / Datasets / Interfaces をホストする Web プラットフォーム |
| jidohub-core | 標準スキーマ・Hub クライアント・config パーサ |
| **jidohub-agents（本リポジトリ）** | Agent をロードして実行する |
| jidohub-datasets | Dataset をロードして `Sample` に正規化する |
| jidohub-interfaces | 実車・シミュレーションとの入出力変換 |

## 依存の原則

- **core にのみ依存する**（星形依存）。datasets / interfaces / server / web に依存しない。
- **`torch` はここで初めて必須依存になる。** core と datasets が torch なしで動くことは
  プラットフォーム全体の前提であり、**その境界が本リポジトリ**である。
- ベンチマークハーネスは datasets を必要とするため optional extra（`[benchmark]`）とし、
  通常のインストールには含めない（**次段階で使用**）。

## インストール

`jidohub-core` は PyPI 未公開のため、**先に core を editable install する**。

```bash
# 1. core を先に入れる（順序が逆だと pip が PyPI を探して失敗する）
pip install -e ../jidohub-core

# 2. agents を入れる（torch はここで入る）
pip install -e '.[dev]'
```

ワークスペース環境では、リポジトリ群のルートで `uv sync` すると core / datasets / agents が
まとめて editable で揃う。

## 使い方（最小例）

```python
from jidohub.agents import AutoAgent

# agent_config.json を読んでクラスを解決し、重みまでロードして返す。
agent = AutoAgent.from_pretrained("acme/CenterPoint-nuScenes@v1", device="cuda:0")

# 入力は単一引数（タスクにより Sample / ImageSample）。出力型はタスクで固定される。
output = agent.predict(sample)
```

Agent の取得・config 検証・スキーマ互換の判定は core（`HubClient` / `load_agent_config`）が担い、
agents は**クラス解決・重みロード・実行**を担う。

画像を入力に取る 2D Agent のために、`import jidohub.agents` 時に既定の画像デコーダを登録する
（既に登録済みなら上書きせず、nvJPEG が利用可能なら優先、なければ Pillow）。

### ストリーミング Agent（tracking / 時系列 E2E）

状態を持つ Agent はストリーミング用のタスク別抽象クラス（`Tracking3DAgent` 等）を継承し、
**`reset()` と `step()` だけを実装する**。系列出力型への集約（`_aggregate`）とメタ情報
（`timestamps` / `ego_to_global`）の収集は抽象クラスと基底が引き受ける。

```python
from jidohub.agents import Tracking3DAgent
from jidohub.core.schemas import Detection3DOutput, Tracking3DInput

class MyTracker(Tracking3DAgent):          # StreamingMixin + BaseAgent
    def reset(self) -> None:
        super().reset()
        self._tracks = {}                  # セッション内の状態を初期化

    def step(self, input: Tracking3DInput) -> Detection3DOutput:
        self._check_initialized()
        ...                                # 現フレームを処理し track_id を採番

# オンライン（interfaces）: reset() してから step() を 1 フレームずつ
# オフライン評価: predict(inputs) が系列をまとめて Detection3DSequence を返す
seq = agent.predict(inputs)               # frames + timestamps + ego_to_global
```

## 現段階の制限

本リポジトリは立ち上げ段階であり、**ネイティブ実装 + inprocess 実行のみ**に対応する。
以下は**次段階**で、要求された場合に明示的なエラー（`NotImplementedError` / `IsolationViolationError`）になる。

- `implementation.type == "remote_code"` の動的ロード（docker runner を要する）
- `runner="docker"` / `transport="shm"`
- ネイティブ実装（CenterPoint / 具体トラッカー等。`NATIVE_AGENTS` は現状空。
  ストリーミング用の**抽象クラス**（`Tracking3DAgent` 等）は提供済み）
- 評価 / ベンチマークハーネス（`[benchmark]` extra は宣言のみ）

未審査コード（`remote_code`）を inprocess で実行することはセキュリティ境界の違反であり、
**黙って inprocess にフォールバックしない**。
