import pandas as pd

# Sample DataFrame
data = {
    'agent_name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Tim White'],
    'agency_name': ['Alice Johnson', 'Smith Group', 'Brown Associates', 'Johnson Agency'],
    'commission_amount': [100, 200, 150, 300]
}

# Create the original DataFrame
df = pd.DataFrame(data)

# Create agent_df by excluding rows where agent_name is equal to agency_name
agent_df = df[df['agent_name'] != df['agency_name']].copy()
agent_df = agent_df[['agent_name'] + [col for col in agent_df.columns if col not in ['agent_name', 'agency_name']]].copy()
agent_df = agent_df.rename(columns={'agent_name': 'producer_name'})
agent_df['producer_type'] = 'Agent'

# Display the new DataFrame
print(agent_df)
