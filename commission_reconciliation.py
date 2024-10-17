import pandas as pd
import os

print("Current Working Directory:", os.getcwd())

# 定义文件路径
files = ['Centene 06.2024 Commission.xlsx', 
         'Emblem 06.2024 Commission.xlsx',
         'Healthfirst 06.2024 Commission.xlsx']


# Parse carrier name from file name
def parse_carrier_name(file_name):
    return file_name.split(' ')[0]

# Print metadata of the table
def analyze_data(df, carrier_name):
    print(f"--- {carrier_name} Data Analysis ---")
    print(f"Number of Records: {len(df)}")
    print(f"Columns:\n {df.columns}")
    print(f"Sample Data:\n {df.head()}\n")

# Centene and Emblem send commision to the agency of the agent,
# whereas Healthfirst table has separate rows for commission to the agent and the belonging agency.
# We noramlize Centene and Emblem tables to matche the format of Healthfirst table.
def separate_agent_and_agency_records(df):
    # Create two dataframes, one for agents and one for agencies
    print(f"Columns:\n {df.columns}")

    agent_df = df[['agent_name'] + [col for col in df.columns if col not in ['agent_name', 'agency_name']]].copy()
    agent_df = agent_df.rename(columns={'agent_name': 'producer_name'})
    agent_df['producer_type'] = 'Agent'
    
    # If a row has same name in agent_name and agency_name, we treat it as agent. To avoid double counting, remove it from agency.
    agency_df = df[df['agent_name'] != df['agency_name']].copy()
    agency_df = agency_df[['agency_name'] + [col for col in agency_df.columns if col not in ['agent_name', 'agency_name']]].copy()
    agency_df = agency_df.rename(columns={'agency_name': 'producer_name'})
    agency_df['producer_type'] = 'FMO'
    
    # Combine agent and agency rows
    combined_df = pd.concat([agent_df, agency_df], ignore_index=True)
    # combined_df = combined_df.reset_index(drop=True)
    analyze_data(combined_df, 'combined')
    return combined_df
    

# Normalize column names
def standardize_columns(df, carrier_name):
    df['carrier'] = carrier_name
    if carrier_name == 'Centene':
        df = df.rename(columns={
                'Writing Broker Name': 'agent_name',
                'Earner Name': 'agency_name',
                'Payment Type': 'enrollment_type', 
                'Pay Period': 'commission_period',
                'Payment Amount': 'commission_amount'})
    elif carrier_name == 'Emblem':
        df = df.rename(columns={
                'Rep Name': 'agent_name',
                'Payee Name': 'agency_name', 
                'Prior Plan': 'enrollment_type', 
                'Effective Date': 'commission_period',
                'Payment': 'commission_amount'})
    elif carrier_name == 'Healthfirst':
        df = df.rename(columns={
                'Producer Name': 'producer_name',
                'Producer Type': 'producer_type', 
                'Enrollment Type': 'enrollment_type',
                'Period': 'commission_period',
                'Amount': 'commission_amount'})

    return df

def normalize_name(name):
    """Normalize the name by lowering case and keeping only first and last names."""
    parts = name.split()
    
    # Keep only the first and last names
    if len(parts) > 2:
        # There are James Martinez Jr and James Gregory Martinez, we treat them as the same person.
        if parts[-1] == 'Jr':
            return ' '.join(parts[:-1])
        return f"{parts[0]} {parts[-1]}"  # First and last name
    return name

def normalize_company_name(name):
    """Normalize company names to a standardized form."""
    standard_names = {
        'delta care corporation': 'Delta Care Corporation'
    }
    normalized_name = name.lower().strip()
    return standard_names.get(normalized_name, name)  # Return original if not found


if __name__ == "__main__":
    
    dfs = {}
    for file in files:
        # Load excel as dataframe
        carrier_name = parse_carrier_name(file)
        dfs[carrier_name] = pd.read_excel(file)

        analyze_data(dfs[carrier_name], carrier_name)
        dfs[carrier_name] = standardize_columns(dfs[carrier_name], carrier_name)

    dfs['Centene'] = separate_agent_and_agency_records(dfs['Centene'])
    dfs['Emblem'] = separate_agent_and_agency_records(dfs['Emblem'])
    # dfs['Emblem'].to_csv('Emblem_after_separate.csv', index=False)

    # dfs['Centene'] = dfs['Centene'].reset_index(drop=True)
    # dfs['Healthfirst'] = dfs['Healthfirst'].reset_index(drop=True)

    for name, df in dfs.items():
        analyze_data(df, name)

    all_commissions = pd.concat(dfs.values(), ignore_index=True)
    

    # Normalize names
    # Broker: List distinct producer_name of broker, identify exclude_list
    broker_producers = all_commissions[all_commissions['producer_type'] == 'Broker']
    distinct_broker_producer_names = broker_producers['producer_name'].unique().tolist()
    distinct_broker_producer_names.sort()
    print("Brokers:\n" + str(distinct_broker_producer_names))
    broker_non_human_names = [name for name in distinct_broker_producer_names if any(keyword in name for keyword in ['DBA', 'LLC', 'Inc', 'Corp', 'Consulting'])]

    # FMO: Mannual find same agency names like ['Delta Care', 'Delta Care Corporation']
    # fmo_producers = all_commissions[all_commissions['producer_type'] == 'FMO']
    # distinct_fmo_producers = fmo_producers['producer_name'].unique().tolist()
    # distinct_fmo_producers.sort()
    # print("FMOs:\n" + str(distinct_fmo_producers))
    
    # If producer type not 'FMO' and it's a human broker, apply normalize_name. Else, apply normalize_company_name
    # We add a new column to keep original information
    all_commissions['normalized_name'] = all_commissions.apply(
        lambda row: normalize_name(row['producer_name']) 
                    if (row['producer_type'] != 'FMO' 
                        and row['producer_name'] not in broker_non_human_names) 
                    else normalize_company_name(row['producer_name']),
        axis=1
    )
    # Normalize commission_period

    # Find top 10
    
    # reorder columns
    front_columns = ['carrier', 'normalized_name', 'producer_name', 'producer_type', 'enrollment_type', 'commission_period', 'commission_amount'] 
    desired_order = front_columns + [col for col in all_commissions.columns if col not in front_columns]  
    all_commissions = all_commissions[desired_order]

    analyze_data(all_commissions, "all")
    all_commissions.to_csv('all_commissions.csv', index=False)




# # 合并所有数据
# all_commissions = pd.concat([emblem_data, centene_data, healthfirst_data], ignore_index=True)

# # 处理代理名字的重复问题（例如去掉中间名等）
# def normalize_name(name):
#     name_parts = name.split()
#     return " ".join([name_parts[0], name_parts[-1]])

# all_commissions['agent_name'] = all_commissions['agent_name'].apply(normalize_name)

# # 计算每个代理或机构的总佣金  
# total_commissions = all_commissions.groupby(['agent_name', 'agency_name']).agg({
#     'commission_amount': 'sum'
# }).reset_index()

# # 找出佣金最高的前 10 名代理或机构
# top_10_performers = total_commissions.sort_values(by='commission_amount', ascending=False).head(10)

# # 输出结果
# print("Top 10 Performers by Commission Payout (June 2024):")
# print(top_10_performers)
# 
# # 将规范化的数据保存为 CSV 文件
# all_commissions.to_csv('normalized_commissions.csv', index=False)
