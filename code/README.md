# Code — HW2 VisDrone video detection

```
code/
├── hw2lib/                          спільний код для локальних ноутбуків (EDA, постпроцесинг)
│   ├── config.py                    класи, протокол, пошук даних, сід
│   ├── data.py                      читання YOLO/ignore, побудова датасету (маска ignore, прорідження, ресайз)
│   ├── evaluate.py                  COCOeval: ignore через iscrowd=1, ≤500 детекцій, mAP при maxDets=500
│   ├── submit.py                    запис і валідація сабміту
│   └── wandb_utils.py               W&B-логін і зображення з боксами
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb   EDA + перевірка конвертації (локально, CPU)
│   └── 02_anchor_free_yolo.ipynb    Kaggle: датасет → перевірка оцінки → тренування → оцінка → сабміт
└── results/
    ├── experiments.csv              усі рани: val-метрики + public LB
    ├── folds_by_clip.csv            5 фолдів по кліпах (EDA §9)
    └── runs/<RUN_ID>/metrics.json   повні метрики кожного рану (Output Kaggle)
```

Kaggle-ноутбуки самодостатні: увесь код (датасет, оцінка, сабміт) — у їхніх клітинках, той самий, що в `hw2lib`.

Формат предиктів: `preds[(clip, frame_stem)] = (xyxy[N,4] у пікселях оригінального кадру, scores[N], cls[N] 0-based)`.

## Запуск

**EDA (локально, CPU).** Дані змагання — у папці над репозиторієм (`hw2/train`, `hw2/val`, `hw2/test`, `hw2/test_images.csv`):
```bash
pip install -r code/requirements.txt
jupyter lab code/notebooks/01_eda_preprocessing.ipynb
```
Графіки зберігаються в `images/eda/`.

**Моделі (Kaggle).** Import Notebook → Input: дані змагання → Accelerator: GPU T4 ×2 → Internet: On →
Secrets: `WANDB_API_KEY`. У першій клітинці вибрати `RUN`, далі Save Version → Save & Run All.
Результати: Output → `artifacts/<RUN_ID>/` (сабміт, `metrics.json`, предикти val/test, ваги).

Версії: ultralytics 8.4.213, torch 2.11.0, Python 3.13 (Kaggle); seed 0.
