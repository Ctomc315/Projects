Numerical_Columns = [
    'SqFt',
    'Seller Concession',
    'Concession %',
    'Original List Price',
    'True $/SqFt',
    'Sold-to-List',
    'Sold to List Price',
    'Days on Market',
    'Culmulative Days on Market',
    'Year Built',
    'True Close Price'
]


df_filtered[Numerical_Columns] = (
    df_filtered[Numerical_Columns]
      .replace({r'[\$,]':'',   # remove dollar signs & commas
                r'%':'',       # remove percent sign
               }, regex=True)
      .apply(pd.to_numeric, errors='coerce')
)


#Feature Engineering here. I think assigning each agent a score of sorts to use for selection would work good. 
df_filtered['Price_per_SqFt'] = df_filtered['True Close Price'] / df['SqFt']

# 1c) Invert the “lower is better” metrics so that **higher** → **better**
df_filtered['DOM_inv'] = df_filtered['Days on Market'].max() - df['Days on Market']


scaler = MinMaxScaler()
to_scale = ['Price_per_SqFt', 'DOM_inv']

df_filtered[to_scale] = scaler.fit_transform(df_filtered[to_scale]) 

weights = {
    'Price_per_SqFt':     0.50,
    'DOM_inv':            0.50,
}

# Compute weighted sum
df_filtered['Agent_Score'] = sum(df_filtered[col] * w for col, w in weights.items())

# Sort agents by descending score
ranked = df_filtered.sort_values('Agent_Score', ascending=False)


def build_recommendation_map(df_filtered):
    # 1) Aggregate each agent’s score within each (City, Zip, Subdivision)
    score_by_loc = (
        df_filtered
        .groupby(['City', 'Zip Code', 'Subdivision', 'List Agent Full Name'])['Agent_Score']
        .mean()
    )
    
    # 2) For each location (City, Zip, Subdivision), find the agent with the highest avg score
    top_agents_map = (
        score_by_loc
        .groupby(level=[0, 1, 2])
        .idxmax()                        # returns tuples like (City, Zip, Subdivision, AgentName)
        .apply(lambda x: x[3])           # pull out the agent name
    )
    
    print("Recommendation map built successfully.")
    return top_agents_map

    def recommend_agent(recommendation_map, city, zip_code, subdivision):
        agent = recommendation_map.loc[(city, zip_code, subdivision)]
        return agent