# CLAUDE.md — jidohub-agents

このファイルには「ソースやconfigを読んでも分からない設計判断」だけを記載する。
依存バージョン・ディレクトリの中身・コマンドの詳細は `pyproject.toml` / 各モジュールを参照すること。

設計の背景と経緯は `docs/design/agents.md` にある。本ファイルは規約のみを扱う。

---

## 1. このリポジトリの位置づけ

jidohub は自動運転向けの Agent / Dataset / Interface 共有プラットフォーム。
**jidohub-agents は Agent のロードと実行を担う**。

| リポジトリ | 役割 |
|---|---|
| jidohub-web | Agents / Datasets / Interfaces をホストする Web プラットフォーム |
| jidohub-core | 標準スキーマ・Hub クライアント・config パーサ |
| **jidohub-agents（本リポジトリ）** | Agent をロードして実行する Python API |
| jidohub-datasets | Dataset をロードして `Sample` に正規化する |
| jidohub-interfaces | 実車・シミュレーションとの入出力変換 |

### 依存の原則

- **core にのみ依存する。** datasets / interfaces / server / web に依存してはならない
- **`torch` はここで初めて必須依存になる。** core と datasets が torch なしで動くことは
  プラットフォーム全体の前提であり、**その境界が本リポジトリ**である
- ベンチマークハーネスは datasets を必要とするため **optional extra** とし、
  通常のインストールには含めない（`[benchmark]`）

### ディレクトリ構成と各ファイルの責務

```
src/
└── jidohub/                     ← namespace package。__init__.py を置かない
    └── agents/
        ├── __init__.py          再エクスポート + 画像デコーダの登録
        ├── exceptions.py        例外型
        ├── base.py              BaseAgent / StreamingMixin / タスク別抽象クラス
        ├── registry.py          ネイティブ実装の明示的レジストリ
        ├── auto.py              AutoAgent（取得 → 解決 → 構築）
        ├── processing.py        PreprocessResult（前後処理の変換情報）
        ├── decoders.py          画像デコーダの登録ポリシー
        ├── native/              ネイティブ実装（純 PyTorch）
        │   └── centerpoint/
        ├── runners/             inprocess / docker
        ├── evaluate/            メトリクス（当面ここに置く）
        └── testing.py           共通テストスイート
```

配置ルール

- **`base.py` に具体的なモデル実装を書かない。** 契約のみを定義する
- **ネイティブ実装は `native/` 配下**。1 モデル 1 サブパッケージ
- **`registry.py` への登録は明示的に行う。** エントリポイント経由の暗黙登録にしない
  （どのクラスが審査済みかが一覧で読めることを優先する。2.4）
- 評価は当面 `evaluate/` に置く。閉ループ評価が視野に入った時点で切り出す

---

## 2. 絶対に守る規約

### 2.1 Agent の契約

- **`predict` の入力は単一引数。** `pack(input)` → RPC → `unpack` → `predict` という
  経路が引数の増加に耐えないため。複合入力が必要なタスクは
  `<TaskName>Input` 型を **core に**定義する（agents 側で独自の入力型を作らない）
- **タスク別の抽象クラスが出力型を固定する。** `Detection3DAgent.predict` は
  `Detection3DOutput` を返す。`TaskType` と出力型の 1 対 1 対応を崩さない
- `BaseAgent` をジェネリック（`Generic[InputT, OutputT]`）にしない。
  `AutoAgent.from_pretrained()` の戻り値は実行時にしか決まらず、
  型引数が `Any` に潰れるため、複雑さに見合わない

### 2.2 `from_pretrained` はテンプレートメソッド

基底が「取得 → config 読込 → クラス解決 → 構築 → 重みロード」の流れを固定し、
**Agent 作者は `_from_config()` と `load_weights()` のみを実装する**。

- `from_pretrained` を **override しないこと**。共通の検証（isolation、
  スキーマ互換、チェックサム）が飛ばされる
- `device` は基底が受け取り `self.device` に保持し、`_from_config` に渡す。
  Agent 作者が環境変数を直接読む実装をしないこと

### 2.3 ステートフル Agent は `StreamingMixin`

- tracking 専用クラス（`BaseTracker`）を作らない。**タスク横断の能力**として定義する
  （`sensing_to_planning` の時系列モデルも同じ機構を使う）
- **継承順序は `class XAgent(StreamingMixin, BaseAgent)` で固定。**
  逆順だと Mixin の `predict` が `BaseAgent` の abstract を上書きできない
- **`StreamingMixin` は単発のタスク別抽象クラス（`Detection3DAgent` / `E2EAgent` 等）と併用しない。**
  単発クラスの `predict(input: Sample)` と Mixin の `predict(inputs: list)`（系列）はシグネチャが非互換で、
  合成すると型検査が正しく矛盾を報告する。ストリーミングは**専用のタスク別抽象クラス**を使う
- **ストリーミング用のタスク別抽象クラスは `base.py` 末尾に定義済み**
  （`Tracking3DAgent` / `Tracking2DAgent` / `InstanceSegmentationTracking2DAgent` / `StreamingE2EAgent`）。
  いずれも `class X(StreamingMixin, BaseAgent)` の組で、対応する系列出力型（`Detection3DSequence` 等）を
  `predict` の戻り値として固定する。`predict` は override せず、戻り値型は `TYPE_CHECKING` ブロックの
  シグネチャ宣言のみで型検査に伝える（既定実装をそのまま使う）
- **`StreamingMixin` を継承したら `predict` は系列を取る。**
- **`_aggregate` はストリーミング用のタスク別抽象クラスが実装する。**
  系列出力型（薄いラッパ）への詰め替えを抽象クラスが引き受けるため、
  **Agent 作者が実装するのは `reset()` と `step()` のみ**。**既定の `predict()` / `_aggregate()` を override しないこと**
- **メタ情報は既定 `predict()` が入力から収集する。** `_collect_meta` が各入力の `.timestamp` を検証して
  系列出力型の `timestamps` に引き写し（`None` や属性欠落は**ループ前に** `StreamingContractError`）、
  `ego_to_global` を全入力が持つ場合のみ `(T,4,4)` にまとめる（2D 系列は `None`）。作者の負担にはしない
- **複合入力型の `detections` 要否・座標系は `predict` の先頭で検証する。**
  `config.requires` があるのに `detections` が `None`、または 3D の `detections` が `EGO` でない場合は
  `UpstreamInputError`（`_check_requirements`）。`step` 直呼び（interfaces）でも同じ検証を呼べるよう protected で公開する
- **不変条件**: `predict(inputs)` の結果は、`reset()` してから `step()` を
  手動ループした結果と一致する。共通テストで機械的に検証する
- 未 `reset()` での `step()` は明示的にエラーにする
  （前シーンの状態が漏れ、track_id が引き継がれて評価結果を汚染する）

### 2.4 未審査コードの実行はセキュリティ境界

- `implementation.type == "remote_code"` の Agent は **inprocess で実行してはならない。**
  core の config 検証で `runtime.isolation == "required"` が強制されているが、
  **agents 側でも二重に検証する**（config を経由しない経路でも破られないようにするため）
- `transport="shm"` は IPC 名前空間の共有を要し隔離が弱まるため、
  `remote_code` の Agent では**拒否する**
- ネイティブ実装のレジストリは `registry.py` の辞書に明示的に列挙する。
  暗黙登録にすると「何が審査済みか」が読めなくなる

### 2.5 ネイティブ実装は純 PyTorch

- **`mmcv` / `mmdet3d` / `spconv` をランタイム依存に入れない。**
  ビルドが不安定で、PyTorch / CUDA バージョンを強く固定するため
- mm 系のモデルは `remote_code` + docker runner の側で扱う。これが二層構造の意味である
- mmdetection3d は**重み工場**として開発環境にのみ置く。
  学習 → 重み変換 → 出力一致テスト、という流れで使う

### 2.6 出力座標を入力の座標系に戻すのは Agent の責務

`2d_tasks.md` 3.2 の規約により、2D 出力の座標は**入力 `Image` の現サイズ基準**でなければならない。
モデル入力用に resize / crop / padding した場合、戻すのは Agent の仕事である。

- **マスクは二値化の前に `F.interpolate` で入力解像度へ戻す。**
  二値化してから最近傍で拡大すると境界が劣化する
- core が `to_source_image()` で scale を伴うマスク移動を `NotImplementedError` に
  しているのは、この再サンプリングを core に持ち込まないため
- 前処理の変換情報は **`PreprocessResult` の戻り値として返す**。
  Agent のインスタンス変数として持ち回ると、ストリーミング時やバッチ時に取り違える

### 2.7 標準スキーマを再定義しない

`Sample` / `Box3D` などの型は core が唯一の正。agents 側で同等の型を定義したり、
フィールドを追加したりしない。表現できない情報が出てきた場合は、
`metadata` に逃がす前に **core のスキーマを拡張すべきかを報告すること**。

### 2.8 画像デコーダの登録は上書きしない

`frame.image.array` を呼ぶにはデコーダの登録が必要だが、core はコーデックに依存しない。
**agents は datasets に依存しない**（星形依存）ため、agents 側にも登録の仕組みが要る。

- `jidohub.agents` の import 時に登録する
- 優先順位: **既に登録済みなら上書きしない** → nvJPEG（利用可能なら）→ Pillow
- 判定は `importlib.util.find_spec` で行い、未インストール時に
  `ImportError` ではなく core の `ImageDecodeError` が出る状態を保つ

---

## 3. 検証の方針

### 3.1 共通テストスイート（`testing.py`）

`BaseAgent` を実装する Agent が満たすべき性質を、**再利用可能な形**で提供する。
ネイティブ実装にも、外部の Agent 作者にも同じ検証を適用できるようにするため。

最低限、以下を含めること。

- 出力型が `TaskType` の宣言と一致する
- `StreamingMixin` の継承順序が正しい
- `predict(inputs)` と `reset()` + `step()` ループの結果が一致する
- 未 `reset()` での `step()` がエラーになる
- `config.prompt.required` が `true` なら、プロンプトなしの入力で明示的にエラー
- 2D なら出力座標が入力 `Image` の現サイズ基準である

### 3.2 ネイティブ実装の三段階検証

| 段階 | 内容 | CI |
|---|---|---|
| 1. モジュール単位 | 各層の出力が参照実装と一致 | fixture 比較。毎 PR |
| 2. end-to-end | 単一サンプルの最終出力が一致 | fixture 比較。毎 PR |
| 3. データセット全体 | NDS / mAP | 手動 / nightly |

- **段階 1・2 の fixture は参照実装の入出力を一度だけ書き出してコミットする。**
  以後の CI は mmdet3d も実データも要求しない
- **段階 1 が通っても段階 2 は必要。** 不一致の原因はモデル本体より
  前処理・後処理（voxel 化の丸め、NMS、座標変換）に潜むことが圧倒的に多い
- 許容基準は段階 3 で NDS ±0.3 ポイント程度。完全一致を追うより、
  段階 1・2 の厳密な一致で実装の正しさを担保する

### 3.3 メトリクスは自前実装しない

nuScenes の NDS / mAP は**公式 devkit の評価コードを使う**。
mmdetection3d も内部で devkit を呼んでいるため、同じコードを使えば
「メトリクス実装の差」という変数が消え、モデル出力の差だけを比較できる。

`evaluate/` の役割は、標準スキーマ → devkit の提出フォーマットへの変換と、
devkit 評価のラッパに留める。

---

## 4. スコープ（当面やらないこと）

以下は**実装せず、必要になった時点で報告して指示を仰ぐこと**。

- **Trainer / 学習ループの標準化**。自動運転モデルの学習はモデルごとの差異が激しく、
  共通抽象の設計はスキーマ設計より難しい。推論の標準化で価値を証明してから
- **CenterPoint voxel 版**（spconv 依存）
- **UniAD のネイティブ実装**。`remote_code` + docker で扱う
- **`pack_into` / 共有メモリ転送**。実車・シミュレーション対応時
- **閉ループ評価**
- **ストリーミング Agent のネイティブ実装**（AB3DMOT / SimpleTrack 等の具体トラッカー）。
  抽象クラス（`Tracking3DAgent` 等）と複合入力型・系列出力型は揃ったが、
  具体実装は該当モデルの要求が来てから着手する

---

## 5. 行動原則

- **契約（`base.py`）の変更は全 Agent に影響する。** 変更を提案する際は、
  既存の実装とテストへの影響を明示すること
- 座標系・単位・配列形状に関わるコードを書くときは、core の docstring と
  `docs/design/tasks/` の規約を先に確認する。**推測で実装しない**
- 参照実装（mmdet3d 等）の挙動が期待と違う場合、**辻褄を合わせず報告する**。
  変換の意味を変える対処は設計判断であり、実装で埋めるべきではない
- 依存ライブラリの追加、core のスキーマで表現できない情報の発見、
  スコープ外（4 章）への進出は自己判断で行わない
