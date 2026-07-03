# 顔認証出席管理システム

Raspberry Piから送信された画像をPCで受信し、YOLOとInsightFaceを使用して顔認証を行うシステムです。

認証に成功すると、Cloud Firestoreに登録されている学生の出席状態を更新します。

## フォルダの用途

| フォルダ | 用途 |
|---|---|
| `models/` | YOLOとInsightFaceのモデル |
| `scripts/` | 顔検出、顔認証、Firestore連携処理 |
| `web/` | 監視画面と顔管理画面 |
| `known_faces/` | 登録した顔写真 |
| `face_embeddings/` | 登録人物の顔特徴量 |
| `images/` | Raspberry Piから受信した画像 |
| `received/` | 最新の認証結果 |
| `uploads/` | 顔登録時の一時ファイル |
| `outputs/` | 顔検出テストの出力画像 |

`known_faces/`、`face_embeddings/`、`images/`、`received/`、`uploads/`、`outputs/` などの実行用フォルダはGitには含まれませんが、プログラムの初回起動時に自動作成されます。

`.venv/`、`.matplotlib/`、`.ultralytics/` も、環境構築やライブラリ実行時に作成されます。

ただし、Firebaseの `scripts/serviceAccountKey.json` は自動生成されないため、利用者が手動で配置する必要があります。

## 機能

- Raspberry PiからJPEG画像を受信
- YOLOによる顔検出（1枚の画像に複数人が写っている場合も検出）
- InsightFaceによる顔認証（検出した顔ごとに照合）
- 最新画像と認証結果をWeb画面に表示
- Web画面から顔写真を登録・削除
- Firestoreの既存memberを出席状態に更新（複数人が認識成功した場合は全員分更新）

顔認証は、新しい画像を受信したときだけ実行されます。

## 認証とDB更新の流れ

Raspberry Piから新しい画像を受信すると、次の順番で処理します。

1. `images/` に受信画像を保存
2. YOLOで画像内の顔を検出
3. 検出した顔ごとにInsightFaceで登録済み顔と照合
4. 類似度がしきい値以上の人物だけ認識成功
5. 認識成功した人物の `members/<student_id>` をFirestoreで更新

2人以上写っている写真でも、認識成功した人物は全員DB更新対象になります。

ただし、YOLOで顔を検出しても、InsightFaceの類似度がしきい値未満の場合はDB更新されません。

現在のしきい値は `scripts/face_embeddings.py` の `DEFAULT_THRESHOLD = 0.5` です。

## 必要なファイル

### YOLOモデル

YOLOの顔検出モデルは自動ダウンロードされないため、次の場所へ手動で配置します。

```text
models/yolov11s-face.pt
```

### InsightFaceモデル

顔認証にはInsightFaceの `buffalo_l` モデルを使用します。

`insightface` パッケージは `requirements.txt` からインストールされ、モデルは初回の顔登録または顔認証時に自動ダウンロードされます。

```text
models/insightface/models/buffalo_l/
```

初回実行時のみインターネット接続が必要です。自動ダウンロードできない環境では、`buffalo_l` モデルを上記の場所へ手動で配置してください。

### Firebaseサービスアカウントキー

Firebaseのサービスアカウントキーを次の場所へ配置します。

```text
scripts/serviceAccountKey.json
```

サービスアカウントキーは秘密情報のため、GitHubなどへアップロードしないでください。

## Firestoreの準備

Firestoreに `members` コレクションを作成し、学生IDをドキュメントIDとして事前に登録します。

例：

```text
members/y-yasukawa
```

顔認証に成功すると、既存ドキュメントの次の項目を更新します。

```text
present: true
lastSeen: サーバー時刻
```

存在しないmemberは新規作成しません。

## 顔写真のファイル名

登録写真は次の形式にします。

```text
<学生ID>_<氏名>.jpg
```

例：

```text
y-yasukawa_安川.jpg
```

この場合、学生IDは `y-yasukawa`、氏名は `安川` です。Firestoreの `members/y-yasukawa` が更新されます。

管理画面の入力欄よりも、アップロードした画像ファイル名が優先されます。

ファイル名が `y-yasukawa_安川.jpg` の場合、登録時に次のように解析されます。

```text
student_id: y-yasukawa
name: 安川
```

ファイル名に `_` がない場合は、管理画面の入力欄の値をもとに登録IDを作成します。

## 導入方法

### macOS

```bash
cd /Users/liqingyang/yolo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows（PowerShell）

```powershell
cd C:\path\to\yolo
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 起動方法

2つのターミナルを使用します。

### 1. Web画面を起動

macOS：

```bash
source .venv/bin/activate
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Windows：

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### 2. 画像受信サービスを起動

macOS：

```bash
source .venv/bin/activate
python rasp.py
```

Windows：

```powershell
.\.venv\Scripts\Activate.ps1
python rasp.py
```

## アクセス先

- 受信画像モニター：<http://127.0.0.1:8000>
- 顔登録・管理画面：<http://127.0.0.1:8000/manage>
- Raspberry Pi送信先：`http://<PCのIPアドレス>:9000/receive_photo`

Raspberry Piからは、`multipart/form-data` の `photo` フィールドでJPEG画像を送信します。

HTMLファイルを直接開かず、必ず上記のHTTPアドレスからアクセスしてください。

## ログの確認

登録時に、ファイル名から抽出したIDと名前がWeb画面側のターミナルに出力されます。

```text
登録ID解析: source=y-yasukawa_安川 person_id=y-yasukawa_安川 student_id=y-yasukawa name=安川
```

Raspberry Piから画像を受信した時は、画像受信サービス側のターミナルに検出人数、認識成功人数、各顔の候補が出力されます。

```text
YOLO検出人数: 2 認識成功人数: 1 threshold: 0.5
顔 0 recognized= True best= ... error= None
顔 1 recognized= False best= ... error= None
```

`YOLO検出人数` は画像内で検出した顔の数です。`認識成功人数` は登録済み顔としきい値以上で一致した人数です。
