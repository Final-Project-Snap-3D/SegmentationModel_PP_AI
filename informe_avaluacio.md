# Informe d'Avaluació de Models de Segmentació
**Dataset:** VizWiz SOD — conjunt de validació (`VizWiz_SOD_test_challenge.json`)  
**Data:** 09/06/2026

---

## 1. Configuració experimental

Els dos models s'han avaluat sobre el mateix conjunt de dades amb el mateix script d'avaluació (`test_evaluation.py`), garantint una comparació justa i reproduïble.

| Paràmetre | Valor |
|---|---|
| Conjunt d'avaluació | `data/test` |
| Anotacions | `VizWiz_SOD_test_challenge.json` |
| Llindar de binarització (UNet/U2Net) | 0.5 (default) |
| Llindar de confiança (YOLO) | 0.25 (default) |

---

## 2. Mètriques d'avaluació

Les mètriques s'han calculat a nivell de píxel comparant la màscara predita amb la màscara *ground truth* de cada imatge:

- **IoU (Intersection over Union):** mesura la superposició entre predicció i GT dividida per la seva unió. És la mètrica estàndard en segmentació. Valors més propers a 1 indiquen millor acord.
- **Dice (F1-score de píxels):** similar a l'IoU però pon doble la intersecció; és més sensible a regions petites.
- **Precision:** proporció de píxels predits com a objecte que realment ho són. Penalitza els falsos positius.
- **Recall:** proporció de píxels reals de l'objecte que el model ha detectat. Penalitza els falsos negatius.

---

## 3. Resultats

### 3.1 U2Net (`best_model_U2Net.pt`)

```
python src/test_evaluation.py \
  --model_path checkpoints/best_model_U2Net.pt \
  --images_dir data/test \
  --annotations .../VizWiz_SOD_test_challenge.json
```

| Mètrica | Valor |
|---|---|
| **IoU** | 0.7765 |
| **Dice** | 0.8397 |
| **Precision** | 0.8366 |
| **Recall** | 0.8987 |

### 3.2 YOLO (`best_yolo.pt`)

```
python src/test_evaluation.py \
  --model_path checkpoints/best_yolo.pt \
  --images_dir data/test \
  --annotations .../VizWiz_SOD_test_challenge.json
```

| Mètrica | Valor |
|---|---|
| **IoU** | 0.8866 |
| **Dice** | 0.9206 |
| **Precision** | 0.9057 |
| **Recall** | 0.9584 |

---

## 4. Comparativa

| Mètrica | U2Net | YOLO | Δ (YOLO − U2Net) | Millora relativa |
|---|---|---|---|---|
| IoU | 0.7765 | **0.8866** | +0.1101 | +14.2 % |
| Dice | 0.8397 | **0.9206** | +0.0809 | +9.6 % |
| Precision | 0.8366 | **0.9057** | +0.0691 | +8.3 % |
| Recall | 0.8987 | **0.9584** | +0.0597 | +6.6 % |

---

## 5. Anàlisi i conclusions

**YOLO supera U2Net en totes les mètriques**, amb una millora especialment significativa en IoU (+14.2 %) i Dice (+9.6 %).

**Qualitat general de segmentació (IoU i Dice):**  
L'IoU del YOLO (0.8866) és notablement superior al d'U2Net (0.7765). Això indica que el YOLO delimita els objectes salients de forma molt més precisa. Un IoU > 0.88 és un resultat excel·lent per a tasques de segmentació sobre imatges "in the wild" com les del dataset VizWiz.

**Falsos positius (Precision):**  
El YOLO presenta una precision de 0.9057 vs 0.8366 d'U2Net. El U2Net tendeix a marcar més píxels com a objecte de forma incorrecta, generant més contorn fals.

**Falsos negatius (Recall):**  
Tots dos models mostren un recall alt (> 0.89), però el YOLO arriba a 0.9584, cosa que significa que gairebé no deixa píxels de l'objecte sense detectar.

**Balanç Precision–Recall:**  
Tots dos models tenen un recall superior a la precision, és a dir, tendeixen a sobre-segmentar lleugerament (marquen alguns píxels de fons com a objecte) en lloc de sub-segmentar. En aplicacions de detecció d'objectes salients per a persones amb discapacitat visual (context VizWiz), és preferible un recall alt — és millor incloure una mica de fons que perdre part de l'objecte d'interès.

**Conclusió:**  
El model YOLO és el millor dels dos per a aquest dataset i tasca. Es recomana usar `best_yolo.pt` com a model de producció, reservant U2Net com a línia base de referència.
