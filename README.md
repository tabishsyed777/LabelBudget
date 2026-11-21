# LabelBudget

Use active learning to choose which text examples a human should label next.

Seed each class with labeled text. Pool entries include hidden labels to simulate an annotator. Each round retrains a multinomial naive Bayes classifier, queries the smallest top-two log-score gap, then reveals that label. The budget counts queries.

## Run locally

From this folder with Python 3.8 or newer, run the included synthetic example and tests. Only the standard library is required.

```bash
python3 app.py
python3 -m unittest -v
```

Copy sample.json and edit its values, then run `python3 app.py your-input.json`.

## Sample output

```json
{
  "queries": [
    {
      "index": 0,
      "text": "storm rain",
      "margin": 0.6931471805599454,
      "revealed": "weather"
    },
    {
      "index": 1,
      "text": "bass solo",
      "margin": 0.4462871026284194,
      "revealed": "music"
    }
  ],
  "training_size": 4
}
```
