#ZAD1#
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

df = pd.read_csv('titanic (1).csv')
df_cleaned = df.drop(columns=['Unnamed: 0'])
encoded_df = pd.get_dummies(df_cleaned)
frequent_itemsets = apriori(encoded_df, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
rules_sorted = rules.sort_values(by="confidence", ascending=False)
print("Najciekawsze reguły asocjacyjne:")
print(rules_sorted[["antecedents", "consequents", "support", "confidence", "lift"]])

plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()