# Code — HW2 VisDrone video detection

```
code/
├── PLAN.md                      план і чеклист етапів
├── requirements.txt
├── hw2lib/                      спільний код для всіх ноутбуків
│   ├── config.py                класи, заморожений протокол, пошук даних, сід
│   ├── data.py                  читання YOLO/ignore, побудова датасету (маска ignore, прорідження, ресайз)
│   ├── evaluate.py              COCOeval (ignore для всіх класів, ≤500 дет.), FPS, збереження артефактів
│   ├── submit.py                запис і валідація сабміту
│   └── wandb_utils.py           логін з Kaggle Secrets, val-кадри з боксами
├── notebooks/
│   └── 01_eda_preprocessing.ipynb   EDA (CPU, локально)
└── results/
    └── experiments.csv          одна таблиця всіх експериментів (val + public LB)
```

Формат предиктів у всьому проєкті:
`preds[(clip, frame_stem)] = (xyxy[N,4] у пікселях оригінального кадру, scores[N], cls[N] 0-based)`.

## Запуск

**Локально (EDA):** покласти дані змагання в `data/` поруч із репо (або задати `DATA_ROOT`), потім
```bash
pip install -r code/requirements.txt
jupyter lab code/notebooks/01_eda_preprocessing.ipynb
```
Графіки зберігаються в `images/eda/`.

**Kaggle (моделі):** Input — дані змагання; Secrets — `WANDB_API_KEY`; Accelerator — GPU T4×2; Internet — ON. Перша клітинка:
```python
!git clone -q https://github.com/PapaJonas/solchanyk_hw2 /kaggle/working/repo
import sys; sys.path.insert(0, "/kaggle/working/repo/code")
```
