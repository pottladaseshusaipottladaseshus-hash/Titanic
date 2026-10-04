import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv")

print("Passenger data:")
print(df.head())

survival_rate = df["Survived"].mean() * 100
print(f"\nOverall survival rate: {survival_rate:.2f}%")

print("\nSurvival rate by gender:")
print(df.groupby("Sex")["Survived"].mean() * 100)

print("\nSurvival rate by passenger class:")
print(df.groupby("Pclass")["Survived"].mean() * 100)

df.groupby("Sex")["Survived"].mean().mul(100).plot(kind="bar")
plt.title("Survival Rate by Gender")
plt.ylabel("Survival Rate (%)")
plt.tight_layout()
plt.savefig("survival_by_gender.png")
plt.show()
