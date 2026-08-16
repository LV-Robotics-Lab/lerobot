# Yuhang Mobile ALOHA migration, 2026-08-14

This record covers the non-Git workstation tree formerly rooted at
`/home/feibo/workspace/aloha_mobile_dummy_lerobot_v3` on
`feibo@yuhang-B850M-C` (`100.64.0.9`).

## Source repositories

- LeRobot workflow, audit, and ROS1 inference source:
  `LV-Robotics-Lab/lerobot` commit `57fbec8`.
- OpenPI PI0.5 Mobile ALOHA LoRA configuration:
  `LV-Robotics-Lab/openpi` commit `e8df129`.
- The source paths and pre-refactor SHA-256 values are listed in
  `PROVENANCE.md`.

## Hugging Face artifacts

All repositories were created private under the `robotics-lv` account.

| Repository                                        | Files |         Bytes | Verification                                               |
| ------------------------------------------------- | ----: | ------------: | ---------------------------------------------------------- |
| `robotics-lv/aloha-mobile-dummy-lerobot-v3`       |    50 |   775,087,951 | dataset parquet local SHA-256 matched Hub LFS              |
| `robotics-lv/pi05-mobile-aloha-lora`              |    10 |     5,206,113 | adapter load succeeded on CUDA                             |
| `robotics-lv/smolvla-mobile-aloha-step1500`       |    24 |   911,616,050 | offline bundle smoke test produced finite `[1, 14]` action |
| `robotics-lv/diffusion-mobile-aloha-step15000`    |     9 | 1,172,918,557 | local policy and processors loaded successfully            |
| `robotics-lv/smolvla-mobile-aloha-expert-history` |    30 | 3,627,008,998 | all four local weight hashes matched Hub LFS               |

The pre-existing private repository `robotics-lv/pi0-attention-audit` retains
the valuable PI0 `E_18K` and `F1_18K` artifacts. Its 17 Hub files total
17,785,040,975 bytes.

Primary weight SHA-256 values:

- PI0.5 LoRA step 20,000:
  `f6f5a3b108ac0166383439e4be4a99c30e12a9813003b5bef25711137bbffb80`
- SmolVLA joint step 1,500:
  `bdcd28f328d4411b92729a3d1152ead9b7c23d0e6bf00b5614d2ede988009162`
- Diffusion Policy step 15,000:
  `4bb2001741d00a85781930ed6cc4b1f41df8e911bbeab005bcf6f4c6b5c1bd63`

The PI0.5 offline cache omitted the
`google/paligemma-3b-pt-224` tokenizer. The adapter and base weights loaded
successfully after the tokenizer was resolved from the Hub; the obsolete local
offline ZIP must not be described as a self-contained deployment package.

## Legacy PI0 disposition

The 216 GB legacy tree contained 26 distinct 8.89 GB weight files across six
historical egg-to-bowl experiment families. No duplicate weight hashes and no
independent checkpoint-selection metrics were found. The retained PI0
artifacts have these SHA-256 values:

- `E_18K`: `d8422a726cab1203623f42106a13eddabb356abe4472da08b1bc64c42c14023e`
- `F1_18K`: `83a065e069a0de5dcfe298c91cef369649f8234e711d7559ea0cf9143b473f9c`

The operator chose selective retention instead of preserving the full legacy
history. A partial Hub archive and a partial NAS transfer were both removed;
the full private archive was not published. The local legacy tree was deleted
after confirming the retained artifacts.

## Workstation cleanup

After the Git source migrations and Hub verification, the remaining 71 GB
workstation tree consisted of upstream checkouts, virtual environments,
downloadable model caches, uploaded dataset/model copies, optimizer states,
and disposable smoke runs. The exact source root
`/home/feibo/workspace/aloha_mobile_dummy_lerobot_v3` was removed on
2026-08-14. It is no longer recoverable from that workstation; retained source
is in the LV Robotics Lab GitHub forks and selected artifacts are in the
private Hugging Face repositories listed above.
