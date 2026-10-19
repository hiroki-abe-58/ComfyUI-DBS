# DBS Soundtrack Separation for ComfyUI（日本語概要）

映画やドラマのような音声を、**音楽**・**効果音**・**セリフ（話し声）** の 3 つに分けるノードです。公式の
**Dynamic Band Splitting (DBS)** モデル（[JusperLee/Dynamic-Band-Splitting](https://github.com/JusperLee/Dynamic-Band-Splitting)、
[論文](https://arxiv.org/abs/2610.14993)）を ComfyUI の中で動かします。

- 動画のセリフを消して音楽と効果音だけを残す、吹き替え用にセリフだけを取り出す、ミックスのバランスを変える、といった用途に使えます。
- モデルは小さく（1.0M / 4.7M パラメータ）、CUDA でも CPU でも動きます。3 つの出力を足すと入力に戻ります。
- ノードは同梱した公式コードを実行するだけで、実行中に何もダウンロードしません。

> **非公式**のノードパックです。DBS の作者や Comfy Org とは関係ありません。英語の [README.md](README.md) が正本です。

## ノード

| ノード | 役割 |
| --- | --- |
| **DBS Loader** | モデル（DBS-TIGER-DnR か DBS-BSRNN-DnR）と実行デバイス（`auto` は ComfyUI の GPU が CUDA ならそれ、ほかは CPU）を選びます |
| **DBS Separate (music / sfx / speech)** | `AUDIO` を受け取り、`music`・`sfx`・`speech`（`AUDIO`）と JSON のレポートを出します |

- 前段には ComfyUI の **Load Audio**（または **Load Video** → **Get Video Components**）をつなぎます。
- セリフを消すには、`music` と `sfx` を ComfyUI の **Merge Audio**（`add`）で足します。同梱のワークフローはそうなっていて、動画用もあります。
- `channels`
  - `mono`：左右の平均を分離します（公式の使い方と同じ）。
  - `each channel`：チャンネルごとに分離し、ステレオのまま出力します（ステレオなら時間は 2 倍）。
- モデルは 44.1 kHz で動きます。ほかのサンプリングレートは ComfyUI 自身のリサンプラーで変換します。
  - `output_sample_rate` = `44100`：モデルのレートのまま出力します。
  - `output_sample_rate` = `input`：入力のレートに戻して出力します。
- 区間の切り方はモデルカードと同じです（DBS-TIGER は 6 秒ずつ 3 秒おき、DBS-BSRNN は 3 秒ずつ 1.5 秒おき）。長さの制限はありません。

| モデル | パラメータ | DnR v2 の SI-SDR（音楽 / 効果音 / セリフ） | RTX 5090 | CPU（24 スレッド） |
| --- | --- | --- | --- | --- |
| **DBS-TIGER-DnR**（既定） | 1.0M | 7.78 / 6.70 / 15.90 dB | 実時間の 16 倍 | 実時間の 0.5 倍 |
| DBS-BSRNN-DnR | 4.7M | 6.48 / 5.73 / 15.01 dB | 実時間の 22 倍 | 実時間の 1.8 倍 |

SI-SDR は作者のモデルカードの値で、ここでは測っていません。速度はここで測った値です（モデル読み込み済み、左右平均）。CUDA の GPU がない場合は DBS-BSRNN が実用的です（DBS-TIGER は CPU だと音声の長さの約 2 倍の時間がかかります）。

## 確認したこと

Windows 11、ComfyUI v0.38.0、RTX 5090 と CPU で確認しました。詳細は [docs/VERIFICATION.md](docs/VERIFICATION.md) にあります。

- **CPU では公式パッケージとビット単位で一致**
  - 対象はプロジェクトページの映画クリップ（Tears of Steel・Sintel、CC BY 3.0）です。
  - 公式 README の使い方（`from_pretrained` と `separate(..., mode="chunked")`）の出力と、6/6 件でバイト単位まで同じでした（両モデル、区間処理のすべての分岐、チャンネルごと）。
  - 公式側は、上流が固定しているバージョンの別環境で動かしました。
- **CUDA**
  - 公式コード自体が、GPU では実行のたびにわずかに違う値を出します。
  - 両側で PyTorch の決定的アルゴリズムを有効にすると、4/4 件でビット単位まで一致しました（30 秒クリップ、両モデル、左右平均とチャンネルごと）。
- **実際の ComfyUI キュー**（CPU と RTX 5090）
  - Load Audio → DBS → Save Audio（FLAC）の出力は、公式の値と 16 ビットの丸めの範囲（半ステップ以内）で一致しました。
  - セリフ除去のワークフロー（動画版を含む）もキュー経由で動きました。
- **クリーンインストール**
  - 新しい ComfyUI v0.38.0 と新しい Python 3.12 環境に、このリポジトリを `git clone` しました。
  - `pip install -r requirements.txt` で追加されたパッケージはありません（`einops` と `safetensors` は ComfyUI に含まれています）。
  - `tools/download_model.py` でモデルをダウンロードし（SHA-256 確認済み）、インストールしたパックで実キュー確認が通りました。
- **CI**（GitHub Actions、Ubuntu と Windows）。次を ComfyUI v0.38.0 と v0.42.1 で確認しています。
  - 実モデルを CPU で動かし、合成音声で公式パッケージと同一プロセス内で比較
  - 同梱コードが固定した上流コミットとバイト単位で同じこと
  - ワークフローが ComfyUI 自身の検証を通ること
  - 実キューで動くこと

**確認していないこと**
- 分離の品質そのもの（このパックは公式の出力を再現するもので、モデルの再評価はしていません）
- 映画のようなミックス以外の音（歌は「音楽」側に入りやすいはずです）
- CI 以外での Linux / macOS、AMD や Apple の GPU（その場合は CPU で動きます）

## インストール

1. V3 ノード API のある ComfyUI（**v0.38.0** 以降）と Python 3.10 以上が必要です。GPU はなくても動きます。
2. このリポジトリを `ComfyUI/custom_nodes/` に clone し、ComfyUI の Python に requirements を入れます（`einops` と `safetensors`。ComfyUI に入っていることが多いです）。

   ```
   git clone https://github.com/hiroki-abe-58/ComfyUI-DBS custom_nodes/ComfyUI-DBS
   python -m pip install -r custom_nodes/ComfyUI-DBS/requirements.txt
   ```

3. モデルをダウンロードします（2 つで 23 MB）。スクリプトはリビジョンを固定し、全ファイルの SHA-256 を確認します。

   ```
   python custom_nodes/ComfyUI-DBS/tools/download_model.py --models-dir models/dbs
   ```

   `models/dbs/DBS-TIGER-DnR/` と `models/dbs/DBS-BSRNN-DnR/` ができ、それぞれに `config.json`・`model.safetensors`・モデルカードの `README.md` が入ります。ほかの場所に置く場合は `extra_model_paths.yaml` のキー `dbs` で指定できます。

4. ComfyUI を再起動し、`workflows/gui/` のワークフローを開きます。
   - `separate_soundtrack.json`：3 つの音を FLAC で保存します。
   - `remove_dialogue.json`：音楽 + 効果音（セリフなしの音）とセリフを、入力のレートのステレオで保存します。
   - `video_remove_dialogue.json`：同じことを動画に対して行い、セリフなしの音を付けた新しい動画として保存します。

   Load Audio / Load Video で自分のファイルを選んでください（ワークフローの初期値は `soundtrack.wav` と `clip.mp4` です）。

## 注意点

- 学習データは DnR v2 です。英語の話し声（LibriSpeech）、音楽（Free Music Archive）、効果音（FSD50K）を映画のように混ぜたものです。ボーカル入りの曲、ほかの言語、音楽がとても大きい素材での結果はご自身で確かめてください。
- 中断（Interrupt）は区間と区間の間で止まります。
- CUDA では、実行のたびに末尾の桁がわずかに変わることがあります（公式コードも同じです）。毎回同じファイルが必要なら、ComfyUI を `--deterministic` 付きで起動してください。

## ライセンス

- このノードパックのコード：**MIT**（[LICENSE](LICENSE)）
- 同梱した DBS のコード（`dbs_comfy/vendor`）：**Apache-2.0**。作者は Bingzhe Zhang・Kai Li・Xiaolin Hu で、一部は MIT ライセンスのプロジェクトからの改変です（[NOTICE](NOTICE)、`dbs_comfy/vendor/UPSTREAM.md`、`dbs_comfy/vendor/THIRD_PARTY.md`）。
- モデル（`tools/download_model.py` を実行したときだけダウンロード）：**Apache-2.0**（モデルカード）。学習データは DnR v2（CC BY 4.0）です。
