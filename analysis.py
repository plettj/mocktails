import matplotlib.pyplot as plt
import pandas as pd

# 1. Load CSV and expand rows by num_people
raw = pd.read_csv("./votes.csv").fillna({"notes1": "", "notes2": "", "notes3": ""})
records = []
for i in [1, 2, 3]:
    records.append(
        pd.DataFrame(
            {
                "meal": i,
                "texture": raw[f"texture{i}"],
                "flavour": raw[f"flavour{i}"],
                "presentation": raw[f"presentation{i}"],
                "overall_score": raw[f"overall{i}"],
                "who_made_it": raw[f"who{i}"],
                "notes": raw[f"notes{i}"],
                "num_people": raw["num_people"],
            }
        )
    )
df = pd.concat(records, ignore_index=True)
df = df.loc[df.index.repeat(df["num_people"].fillna(1).astype(int))].reset_index(
    drop=True
)

# 2. Map meal names and calculate total
meal_names = {1: "Quesadillas", 2: "Mushroom Stew", 3: "Egg Tomato Stir Fry"}
df["meal_name"] = df["meal"].map(meal_names)
df["total"] = df[["texture", "flavour", "presentation", "overall_score"]].sum(axis=1)

# 3. Bar chart for the four metrics, ordered and colored
metrics = ["texture", "flavour", "presentation", "overall_score"]
order = ["Quesadillas", "Mushroom Stew", "Egg Tomato Stir Fry"]
avg_scores = df.groupby("meal_name")[metrics].mean().reindex(order)

x = range(len(metrics))
width = 0.25
fig, ax = plt.subplots()
for idx, meal in enumerate(order):
    ax.bar([p + idx * width for p in x], avg_scores.loc[meal], width, label=meal)
ax.set_xticks([p + 1.5 * width for p in x])
ax.set_xticklabels(metrics)
ax.set_ylabel("Average Score")
ax.legend()
plt.tight_layout()
plt.show()

# 4. Print text summaries of comments per dish
for meal, group in df.groupby("meal_name"):
    comments = [c for c in group["notes"] if isinstance(c, str) and c.strip()]
    print(f"\n{meal} comments:")
    for note in comments:
        print(f" - {note}")

# 5. Polarization analysis (variance) on all five metrics
pol_metrics = ["texture", "flavour", "presentation", "overall_score", "total"]
variances = df.groupby("meal_name")[pol_metrics].var()
print("\nMost polarizing dish by metric:")
for m in pol_metrics:
    dish = variances[m].idxmax()
    print(f" {m}: {dish} (variance={variances[m].max():.2f})")

# 6. Guessing accuracy and mix-up analysis
correct_map = {1: "j", 2: "p", 3: "m"}
df["correct_guess"] = df["who_made_it"] == df["meal"].map(correct_map)
accuracy = df.groupby("meal_name")["correct_guess"].mean() * 100
print("\nGuess accuracy by dish:")
for meal, acc in accuracy.items():
    print(f" {meal}: {acc:.1f}% correct")
errors = df.loc[~df["correct_guess"], "who_made_it"].value_counts(normalize=True) * 100
if not errors.empty:
    common_error = errors.idxmax()
    print(
        f"\nMost common wrong guess: '{common_error}' ({errors.max():.1f}% of errors)"
    )
