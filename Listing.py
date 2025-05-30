import pandas as pd;
import matplotlib.pyplot as plt;
import seaborn as sns;
import plotly.express as px;
import os;

os.makedirs("visualizations", exist_ok=True)

df = pd.read_csv("listings.csv", encoding="latin1")
print(df.head())

df.columns = [col.strip().lower() for col in df.columns]
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())

#check missing values
print("missing values",df.isnull().sum())

df = df.dropna(subset=['price'])
print("price values before:",df['price'])


#Drop coloumns with >30% missing data
df=df.dropna(thresh=0.7*len(df),axis=1)
print("price values after:",df['price'])

#Fill missing numerical values using median
df['reviews_per_month'] = df['reviews_per_month'].fillna(df['reviews_per_month'].median())


df = df[df['price'] <= 500]  # Filter
print("price values after filter:",df['price'])

# Create a new feature: "price_per_person" (assuming 2 guests)
df['price_per_person'] = df['price'] / 2

# Convert 'last_review' to datetime
df['last_review'] = pd.to_datetime(df['last_review'])

print(df.describe())

# Top 5 most expensive neighborhoods
print(df.groupby('neighbourhood')['price'].median().sort_values(ascending=False).head(5))

plt.figure(figsize=(10, 6))
sns.heatmap(df[['price', 'minimum_nights', 'availability_365', 'number_of_reviews']].corr(), annot=True)
plt.title("Correlation Matrix")
plt.savefig('visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close() 


#price dustribution by neighborhood
plt.figure(figsize=(12, 6))
sns.boxplot(x='neighbourhood', y='price', data=df, showfliers=False)  # Hide outliers
plt.xticks(rotation=90)
plt.title("Price Distribution by Neighborhood")
plt.savefig('visualizations/price_by_neighborhood.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Reviews over time
plt.figure(figsize=(10, 6))
df.set_index('last_review')['number_of_reviews'].resample('M').count().plot()
plt.title("Monthly Reviews Trend")
plt.savefig('visualizations/monthly_reviews_trend.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# Average price by room type and neighborhood
pivot = pd.pivot_table(df, values='price', index='neighbourhood', columns='room_type', aggfunc='median')
sns.heatmap(pivot, cmap="YlGnBu")
plt.title("Median Price by Neighborhood & Room Type")
plt.savefig('visualizations/price_by_neighborhood_roomtype.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


sns.scatterplot(x='price', y='availability_365', data=df, hue='room_type')
plt.title("Price vs. Availability")
plt.savefig('visualizations/price_vs_availability.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
       
df.to_csv("cleaned_berlin_airbnb.csv", index=False)

print("All visualizations saved to the 'visualizations' folder!")