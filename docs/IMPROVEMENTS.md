# 向上施策 — OOD 汎化・破滅的忘却への対策

本ドキュメントは、レポートの主要知見を踏まえて実装した 3 つの向上施策の設計・根拠・実行方法をまとめたものです。

## 背景:現状の課題

レポート(`report.pdf`)で得られた知見:

1. **Progressive は Hard で最高性能(0.97±0.02)だが、Easy/Medium を破滅的に忘却する**(成功率 0.00 / 0.02)
2. **OOD 転移は全戦略でほぼゼロ**。信頼できる最良の転移は多タスクベースライン(Mixed/Random, 0.059±0.030)
3. 結論として **「順序」ではなく「多様性」が OOD 転移を駆動する**

この 3 点に直接対応する施策を実装しました。

## 施策 1: Interleaved Replay カリキュラム(`progressive_replay`)

**狙い:破滅的忘却の解消(知見 1)**

Progressive の段階構造は維持しつつ、各エピソード開始時に確率 `replay_prob`(既定 0.2)で過去ステージの環境を再訪(リハーサル)します。継続学習分野で標準的な rehearsal 手法のカリキュラム版で、EWC(重み空間の正則化、探索実験で効果薄)と異なりデータ空間で忘却を防ぎます。

設計上の要点:

- **リプレイエピソードは昇格判定ウィンドウから除外**します。Easy での成功が現ステージの成功率を水増しして早すぎる昇格を招くのを防ぐためです(`curriculum.py` の `_is_replay_episode`)。
- 期待される効果:Hard 性能をほぼ維持したまま Easy/Medium の成功率を回復し、訓練分布の実効多様性が増えることで OOD 転移も Mixed 水準に近づく可能性があります。

```bash
python train.py --strategy progressive_replay --seeds 0 1 2 3 4 5 6 7 8 9
```

ハイパーパラメータ: `configs/default.yaml` の `training.replay_prob`(既定 0.2)

## 施策 2: 段階内環境プール(`--diverse`)

**狙い:訓練多様性の増大による OOD 転移向上(知見 2・3)**

「多様性が順序に勝る」なら、多様性そのものを増やすのが最短の向上施策です。各難易度ティアを単一環境から**構造的に異なる環境のプール**に拡張し、エピソードごとにティア内からサンプリングします:

| ティア | プール |
|---|---|
| easy | Empty-8x8, Empty-Random-6x6, Empty-16x16 |
| medium | FourRooms, Unlock |
| hard | KeyCorridorS3R1, KeyCorridorS3R2 |

**OOD 評価の妥当性を守るため、テスト/転移環境ファミリー(DoorKey, MultiRoom, LavaCrossing, SimpleCrossing, DistShift)はプールに含めていません**。ゼロショット評価はゼロショットのままです。

```bash
# 多様性 × 順序の分離実験:最有力は mixed --diverse と progressive_replay --diverse
python train.py --strategy mixed --diverse --seeds 0 1 2 3 4 5 6 7 8 9
python train.py --strategy progressive_replay --diverse --seeds 0 1 2 3 4 5 6 7 8 9
```

モデルは `mixed_div_seed*.zip` のように `_div` サフィックスで保存されます。

## 施策 3: 観測ノイズ正則化(`--obs-noise`)

**狙い:観測レベルの過学習抑制(知見 2)**

FlatObsWrapper の平坦化 one-hot 観測は訓練環境固有のパターンを丸暗記しやすく、OOD での表現破綻の一因と考えられます。訓練時のみ観測にガウスノイズ(σ = `obs_noise_std`、既定 0.05)を加える入力正則化を追加しました(`envs/wrappers.py` の `ObsNoiseWrapper`)。評価時はクリーンな観測のままです。

```bash
python train.py --strategy mixed --obs-noise --seeds 0 1 2 3 4 5 6 7 8 9
```

モデルは `_noise` サフィックスで保存されます(`--diverse` と併用可: `_div_noise`)。

## 推奨実験マトリクスと評価

主張したい比較ごとに最小限の条件:

| 条件 | 検証すること |
|---|---|
| `progressive_replay` vs `progressive` | リプレイで忘却が解消するか(forgetting 分析) |
| `mixed --diverse` vs `mixed` | ティア内多様性の追加で OOD 転移が伸びるか |
| `progressive_replay --diverse` vs 上記2つ | 順序+リハーサル+多様性の複合効果 |
| `mixed --obs-noise` vs `mixed` | 入力正則化単体の寄与 |

評価は既存パイプラインがそのまま使えます(新戦略・バリアントは `evaluate.py` の choices に登録済み):

```bash
python evaluate.py --strategy progressive_replay --seed 0
python evaluate.py --strategy mixed_div --seed 0
python evaluate.py --all --n-eval 50
python analysis/plot_forgetting.py --n-eval 20
```

## 実装ファイル一覧

- `curriculum.py` — `progressive_replay` 戦略、`ENV_POOLS`、リプレイ除外ロジック
- `envs/wrappers.py` — `ObsNoiseWrapper`
- `train.py` — `--diverse` / `--obs-noise` フラグ、タグ命名(`_div` / `_noise`)
- `configs/default.yaml` — `replay_prob`、`obs_noise_std`、`variant_strategies`
- `evaluate.py` — バリアント戦略の評価対応
