# interactive-grouping

Jupyter-віджет для перегляду і **ручного редагування групувань (binning / WoE)** поверх
[optbinning](https://github.com/guillermo-navas-palencia/optbinning). Інтерфейс за мотивами
SAS Enterprise Miner *Interactive Grouping*: дрібні (fine) біни → групи, таблиця Fine/Coarse,
статистика Original / Applied / New, роль змінної.

Лише Python + популярні бібліотеки: `optbinning`, `ipywidgets`, `plotly` (або `matplotlib`),
`pandas`, `numpy`. Без JavaScript-збірки, тож працює у JupyterLab / Notebook / VS Code /
Google Colab.

## Швидкий старт

```python
from optbinning import BinningProcess
from interactive_grouping import InteractiveBinning

bp = BinningProcess(variables, categorical_variables=cats, special_codes=[-1]).fit(X, y)
w = InteractiveBinning(bp, X, y)        # backend="matplotlib" якщо plotly не рендериться
w                                       # показати віджет

# після редагування
bp_edited = w.to_binning_process()      # звичайний fitted optbinning.BinningProcess
X_woe     = w.transform(X)              # WoE з поточними групуваннями
w.save_json("splits.json")              # зберегти ручну роботу; w.load_json(...) — відновити
```

Демо: `notebooks/demo_interactive_grouping.ipynb` (синтетичний кредитний датасет, без завантажень).

## Що вміє віджет

**Вкладка Groupings**

* Графік дрібних бінів: стовпці кольору групи, смуги та номери груп, межі груп на осі X,
  червоні відрізки — WoE групи, тонка лінія — WoE дрібних бінів. Поруч — графік груп
  (кількість non-event/event + лінія WoE).
* **Detail Level = Coarse** — рядок на групу, **лише перегляд** (жодних змін).
* **Detail Level = Fine** — рядок на дрібний бін (числові: ~N квантилів, межі груп завжди
  входять у дрібні біни) або на категорію. Тут відбувається все редагування: виділіть рядок →
  у **Assign selected to group** виберіть групу → **Assign**. Пропонуються лише допустимі варіанти.
  * **Числові змінні** (групи завжди суміжні, монотонність не ламається):
    * *попередня група* — переходять усі дрібні біни від початку групи до виділеного включно;
    * *наступна група* — виділений і всі наступні дрібні біни до кінця групи;
    * *нова група — this and higher / this and lower* — група ділиться в цьому місці;
    * якщо група спорожніла, вона просто зливається з сусідньою;
    * **Split selected fine bin** — ділить дрібний бін на два (значення за замовчуванням —
      медіана); групування при цьому не змінюється.
  * **Категоріальні змінні** — кожна категорія є дрібним біном; її можна перенести в будь-яку
    іншу групу або виділити в нову. Якщо група спорожніла, групи зливаються.
  Після перерахунку групи перенумеровуються (числові — за порядком значень, категоріальні —
  optbinning сортує за event rate); фактичний номер показується в повідомленні.
* **Variable Statistics**: Groups / IV / Gini / тренд / сила IV для *Original* (optbinning),
  *Applied* (застосоване) і *New* (поточне редагування).
* **Variable Role**: Input / Rejected (Rejected не потрапляє в `to_binning_process()` і `transform()`).
* **Apply · Discard · Undo · Redo · Reset variable · Reset All Changes**. Нове групування
  не застосовується до натискання **Apply**; незастосовані зміни зберігаються для кожної змінної.
* **Advanced** — лише з `InteractiveBinning(..., advanced=True)` (за замовчуванням прихований):
  кількість дрібних бінів, ручний список меж, оптимізатор optbinning з обмеженнями (тренд,
  max groups, min group %), повна binning table. ⚠ **Експериментально, не перевірено в реальній
  роботі — рекомендується перевіряти отримані групування.**

**Вкладка Variables** — усі змінні: роль, тип, групи / IV / Gini (original vs new), тренд,
що змінено; збереження/завантаження JSON.

## Як це працює (чому статистики «справжні»)

Кожен перерахунок — це `optbinning.OptimalBinning(user_splits=..., user_splits_fixed=[True]*n)`
з тими ж параметрами, що й у вихідному групінгу (special codes, cat_unknown, ...), але з вимкненими
обмеженнями, які конфліктують із фіксованими розбиттями (`monotonic_trend`, `max_n_bins`, `min_bin_size`, ...).
Тому WoE/IV/таблиця рахує сам optbinning, а `w.to_binning_process()` відтворює точно ті самі біни
(перевірено тестами, включно з `transform`).

Обмеження optbinning, які варто знати:
* розбиття, що дає «чистий» бін (лише події або лише не-події, або порожній), optbinning відхиляє —
  віджет покаже зрозуміле повідомлення;
* мінімум 2 групи на змінну;
* Missing / Special завжди окремі біни (optbinning не дозволяє приєднати їх до групи, на відміну від SAS);
* підтримується **бінарний** таргет (`OptimalBinning`). Continuous/multiclass — не реалізовано.

## Архітектура

```
interactive_grouping/
  session.py        # BinningSession: набір змінних, ролі, експорт (BinningProcess, WoE, JSON)
  variable.py       # VariableBinning: перерахунок через optbinning, історія, fine-рівень
  strategies.py     # GroupingStrategy: NumericalStrategy / CategoricalStrategy (патерн Strategy)
  operations.py     # чисті функції редагування розбиттів (merge, cutoffs, fine → groups)
  history.py        # узагальнена історія undo / redo
  metrics.py        # WoE, тренд event rate, сила IV
  plotting.py       # FigureBackend: PlotlyBackend / MatplotlibBackend
  ui_components.py  # CSS, картки, HTML-таблиці, рядки таблиці груп (без стану)
  widget.py         # InteractiveBinning: контролер UI
  exceptions.py, _utils.py, data.py
notebooks/demo_interactive_grouping.ipynb
tests/              # pytest за модулями + "headless" віджет (симуляція кліків) для обох бекендів
```

Принципи:

* **Single responsibility** — модель (`session`, `variable`), чисті операції (`operations`),
  представлення (`ui_components`, `plotting`) і контролер (`widget`) розділені.
* **Open/closed + Strategy** — усе, що відрізняється для числових і категоріальних змінних,
  інкапсульовано в `GroupingStrategy`; новий тип змінної = новий клас без змін у викликах.
* **Dependency inversion** — віджет залежить від абстракції `FigureBackend`, а не від plotly /
  matplotlib; бекенд створюється через `get_backend(name)`.
* **DRY** — спільна підготовка даних графіків для обох бекендів, один декоратор `_action` для
  обробки помилок / рівня деталізації в усіх callback'ах, спільні колонки таблиць.

### Стиль коду

PEP 8, docstrings за PEP 257 у numpydoc-стилі (як в optbinning), type hints. Перевірка:

```bash
ruff check .        # pycodestyle, pyflakes, isort, pep8-naming, pydocstyle, bugbear, ...
pytest
```

## Встановлення / середовище

```bash
pip install -e .            # або: pip install -r requirements.txt
pip install -e .[dev]       # + jupyterlab, pytest, ruff, matplotlib
pytest                      # 72 тести
```

### Google Colab

Варіант 1 — архів: локально
`python -c "import shutil; shutil.make_archive('interactive_grouping', 'zip', '.', 'interactive_grouping')"`,
завантажте `interactive_grouping.zip` у Colab і виконайте:

```python
!pip -q install optbinning
!unzip -o -q interactive_grouping.zip
from interactive_grouping import InteractiveBinning
```

Варіант 2 — якщо проєкт у git-репозиторії: `!pip install git+https://<repo-url>`.

> Colab не тестувався в цьому середовищі.

## Рендер графіків

Графіки **не** малюються через `ipywidgets.Output` (його оновлення з callback'ів ненадійне поза
JupyterLab — PyCharm, VS Code). Замість цього оновлюється звичайна властивість віджета:

* `backend="plotly"` (за замовчуванням) — інтерактивна фігура в `<iframe srcdoc>` усередині
  `widgets.HTML`; plotly.js вантажиться з CDN (потрібен інтернет). Офлайн: `plotly_js="inline"`
  (працює, але кожна фігура несе ~3.5 MB plotly.js).
* `backend="matplotlib"` — PNG у `widgets.Image`; не потребує інтернету, найлегший варіант.
