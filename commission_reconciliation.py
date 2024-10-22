import pandas as pd
import os
import csv 

print("Current Working Directory:", os.getcwd())

# Define the list of Excel files to be processed
files = ['Centene 06.2024 Commission.xlsx', 
         'Emblem 06.2024 Commission.xlsx',
         'Healthfirst 06.2024 Commission.xlsx']

# Dictionary to store the mapping of original names to normalized names
normalized_name_log = {}

def load_data(file_name):
    """Load data from an Excel file into a DataFrame."""
    try:
        df = pd.read_excel(file_name)
        print(f"Successfully loaded data from {file_name}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")
        return None
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None

def parse_carrier_name(file_name):
    """Extract and return the carrier name from the file name."""
    return file_name.split(' ')[0]

# Print metadata of the table
def analyze_data(df, carrier_name):
    """Print the number of records, column names, and sample data."""
    try:
        print(f"--- {carrier_name} Data Analysis ---")
        print(f"Number of Records: {len(df)}")
        print(f"Columns:\n {df.columns}")
        print(f"Sample Data:\n {df.head()}\n")
    except Exception as e:
        print(f"Error analyzing data for {carrier_name}: {e}")

# Centene and Emblem send commision to the agency of the agent,
# whereas Healthfirst table has separate rows for commission to the agent and the belonging agency.
# We noramlize Centene and Emblem tables to matche the format of Healthfirst table.
def separate_agent_and_agency_records(df):
    """Separate agent and agency records and return a combined DataFrame."""
    # print(f"Columns:\n {df.columns}")
    
    # Create two dataframes, one for agents and one for agencies
    agent_df = df[['agent_name'] + [col for col in df.columns if col not in ['agent_name', 'agency_name']]].copy()
    agent_df = agent_df.rename(columns={'agent_name': 'producer_name'})
    agent_df['producer_type'] = 'Agent'
    
    # If a row has same name in agent_name and agency_name, we treat it as agent to avoid double counting
    agency_df = df[df['agent_name'] != df['agency_name']].copy()
    agency_df = agency_df[['agency_name'] + [col for col in agency_df.columns if col not in ['agent_name', 'agency_name']]].copy()
    agency_df = agency_df.rename(columns={'agency_name': 'producer_name'})
    agency_df['producer_type'] = 'FMO'
    
    # Combine agent and agency rows
    combined_df = pd.concat([agent_df, agency_df], ignore_index=True)
    analyze_data(combined_df, 'combined')
    return combined_df

def standardize_columns(df, carrier_name):
    """Standardize column names according to the specific carrier's format."""
    try:
        df['carrier'] = carrier_name  # Add carrier name to the DataFrame
        # Normalize column names based on carrier type
        if carrier_name == 'Centene':
            df = df.rename(columns={
                    'Writing Broker Name': 'agent_name',
                    'Earner Name': 'agency_name',
                    'Payment Type': 'enrollment_type', 
                    'Pay Period': 'commission_period',
                    'Payment Amount': 'commission_amount', 
                    # Not requested to be listed
                    'Centene ID': 'member_id',
                    'Member Name': 'member_name',
                    'Effective Date': 'effective_date',
                    'Member Term Date': 'term_date',
                    'Plan Name': 'plan_name'
                    })
        elif carrier_name == 'Emblem':
            df = df.rename(columns={
                    'Rep Name': 'agent_name',
                    'Payee Name': 'agency_name', 
                    'Prior Plan': 'enrollment_type', 
                    'Effective Date': 'commission_period',
                    'Payment': 'commission_amount', 
                    # Not requested to be listed
                    'Member ID': 'member_id',
                    'Member First Name': 'member_first_name',
                    'Member Last Name': 'member_last_name',
                    'Effective Date': 'effective_date',
                    'Term Date': 'term_date',
                    'Plan': 'plan_name'})
        elif carrier_name == 'Healthfirst':
            df = df.rename(columns={
                    'Producer Name': 'producer_name',
                    'Producer Type': 'producer_type', 
                    'Enrollment Type': 'enrollment_type',
                    'Period': 'commission_period',
                    'Amount': 'commission_amount', 
                    # Not requested to be listed
                    'Member ID': 'member_id',
                    'Member Name': 'member_name',
                    'Member Effective Date': 'effective_date',
                    'Disenrolled Date': 'term_date',
                    'Product': 'plan_name'})
        return df
    except KeyError as e:
        print(f"Error normalizing data: Missing column {e}")
        return df
    except Exception as e:
        print(f"Unexpected error during normalization: {e}")
        return df

def log_normalized_names(original_name, normalized_name):
    """Log and store the original name and its corresponding normalized name, if they differ."""
    if normalized_name not in normalized_name_log:
        normalized_name_log[normalized_name] = set()  # Initialize a set if not present
    normalized_name_log[normalized_name].add(original_name)  # Add the original name to the set

def handle_null_producer_types(all_commissions):
    """Analyze and replace null values in the producer_type column."""
    null_producer_type_rows = all_commissions[all_commissions['producer_type'].isnull()]
    analyze_data(null_producer_type_rows, "abnormal data null_producer_type")
    # Replace all null values in the producer_type column with 'FMO', after checking the existing data
    # because all entries without a producer_type belong to Delta Care Corporation from Healthfirst carrier
    all_commissions['producer_type'] = all_commissions['producer_type'].fillna('FMO')
    return all_commissions

def normalize_name(name):
    """Normalize the name by lowering case and keeping only first and last names."""
    parts = name.split()
    
    # Keep only the first and last names
    if len(parts) > 2:
        # There are James Martinez Jr and James Gregory Martinez, we treat them as the same person.
        if parts[-1] == 'Jr':
            normalized = ' '.join(parts[:-1])
        # An agent may not be a person, for example, Kirk Baker DBA Carter-Thomas is an agent.
        elif name in agent_non_human_names:
            normalized = name
        else:
            normalized = f"{parts[0]} {parts[-1]}"  # Keep first and last name
    else:
        normalized = name  # Return name as is if it has only first and last names
    log_normalized_names(name, normalized)
    return normalized

def normalize_company_name(name):
    """Normalize company names to a standardized form."""
    standard_company_names = {
        'delta care corporation': 'Delta Care Corporation',
        'delta care': 'Delta Care Corporation'
    }
    normalized = standard_company_names.get(name.lower().strip(), name)  # Normalize based on the mapping
    log_normalized_names(name, normalized)
    return normalized  # Return original if not found

def save_normalized_names_to_csv(file_name):
    """Save the original and normalized name mappings to a CSV file."""
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Original Name', 'Normalized Name'])  # Write header
        for original, normalized in normalized_name_log.items():
            writer.writerow([original, normalized])  # Write each mapping
    print(f"Normalized name mappings saved to {file_name}")

def normalize_commission_period(period):
    """Normalize commission_period date to YYYY-MM format."""
    try:
        # Use pd.to_datetime to parse and convert dates automatically
        date = pd.to_datetime(period, errors='coerce')
        if pd.notna(date):
            return date.strftime('%Y-%m')  # Return in YYYY-MM format
        
        return pd.to_datetime(period, format='%b-%y').strftime('%Y-%m')
    except:
        return pd.NaT  # Return NaT for unparseable values

def normalize_date(date):
    """Normalize date to YYYY-MM-DD format."""
    try:
        # Convert to datetime
        date_value = pd.to_datetime(date, errors='coerce')
        if pd.isna(date_value): 
            return date_value  # Return NaT as is

        return date_value.strftime('%Y-%m-%d')  # Format as YYYY-MM-DD
    except Exception as e:
        print(f"Error normalizing date: {e}")
        return None  # Return NaT for unparseable values

def normalize_enrollment_type(enrollment_type, prior_plan):
    """Normalize enrollment type based on specific rules."""
    if enrollment_type in ['Monthly Renewal']:
        return 'Renewal - Monthly' 
    elif enrollment_type in ['Adjustment']:
        return 'Adjustment'
    elif enrollment_type in ['Yes']:  # 'Yes' indicates prior plan in Emblem table
        return 'Prior Plan Exists'

    return enrollment_type  # Return as is if no match

def create_full_name(df, first_name_col, last_name_col, new_name_col):
    """Create a new column with full names by combining specified first and last name columns. """
    df[new_name_col] = df[first_name_col] + ' ' + df[last_name_col]
    return df

def get_distinct_producer_names(df, producer_type):
    """Get a sorted list of distinct producer names for the given producer type."""
    distinct_names = df[df['producer_type'] == producer_type]['producer_name'].unique().tolist()
    distinct_names.sort()
    # print(f"{producer_type}:\n{distinct_names}")
    return distinct_names

def identify_non_human_names(distinct_names):
    """Identify non-human names from the list of distinct names based on specific keywords."""
    keywords = ['DBA', 'LLC', 'Inc', 'Corp', 'Consulting']
    non_human_names = [name for name in distinct_names if any(keyword in name for keyword in keywords)]
    return non_human_names

def normalize_producer_names(all_commissions):
    """Normalize producer names based on their type."""
    # Get distinct names for agents and identify non-human names after mannual searching
    distinct_agent_names = get_distinct_producer_names(all_commissions, 'Agent')
    agent_non_human_names = identify_non_human_names(distinct_agent_names)
    
    # Get distinct names for brokers and identify non-human names after mannual searching
    distinct_broker_names = get_distinct_producer_names(all_commissions, 'Broker')
    broker_non_human_names = identify_non_human_names(distinct_broker_names)
    
    # FMO normalization: Mannual find same agency names like ['Delta CARE CORPORATION', 'Delta Care', 'Delta Care Corporation']
    # So we can set the standard_company_names in normalize_company_name function
    distinct_fmo_names = get_distinct_producer_names(all_commissions, 'FMO')

    return agent_non_human_names, broker_non_human_names
    
if __name__ == "__main__":
    dfs = {}  # Dictionary to hold DataFrames for each carrier

    # Load and process each commission file
    for file in files:
        carrier_name = parse_carrier_name(file)
        df = load_data(file)
        if df is not None:
            # Standardize columns and analyze data
            dfs[carrier_name] = standardize_columns(df, carrier_name)
            analyze_data(df, carrier_name)

    # Separate agent and agency records for specific carriers
    dfs['Centene'] = separate_agent_and_agency_records(dfs['Centene'])
    dfs['Emblem'] = separate_agent_and_agency_records(dfs['Emblem'])

    # Analyze data again for all carriers
    for name, df in dfs.items():
        analyze_data(df, name)

    # Normalize full name columns for Emblem
    dfs['Emblem'] = create_full_name(dfs['Emblem'], 'member_first_name', 'member_last_name', 'member_name')

    # Concatenate all commission data into a single DataFrame
    all_commissions = pd.concat(dfs.values(), ignore_index=True)

    # Handle null producer types
    all_commissions = handle_null_producer_types(all_commissions)

    # Normalize producer names and obtain non-human names for agents and brokers
    agent_non_human_names, broker_non_human_names = normalize_producer_names(all_commissions)
    
    # If producer type not 'FMO' and it's a human broker, apply normalize_name. Else, apply normalize_company_name
    # We add a new column to keep original information
    all_commissions['normalized_name'] = all_commissions.apply(
        lambda row: normalize_name(row['producer_name']) 
                    if (row['producer_type'] != 'FMO' 
                        and row['producer_name'] not in broker_non_human_names) 
                    else normalize_company_name(row['producer_name']),
        axis=1
    )

    # Save the name mappings to a CSV file to make the information easily accessible for checking and future use
    # save_normalized_names_to_csv('normalized_name_mappings.csv')

    # Normalize commission period and date columns
    all_commissions['commission_period'] = all_commissions['commission_period'].apply(normalize_commission_period)
    all_commissions['effective_date'] = all_commissions['effective_date'].apply(normalize_date)
    all_commissions['term_date'] = all_commissions['term_date'].apply(normalize_date)

    # Select and reorder the columns
    front_columns = ['carrier', 'normalized_name', 'producer_name', 'producer_type', 'enrollment_type', 'commission_period', 'commission_amount'] 
    other_columns = ['member_id', 'member_name', 'effective_date', 'term_date', 'plan_name']
    desired_order = front_columns + other_columns
    all_commissions = all_commissions[desired_order]
 
    # Store final data
    all_commissions.to_csv('all_commissions.csv', index=False)

    # Caculate the total commissions each agent and agency got for commission period June 2024.
    june_2024_data = all_commissions[all_commissions['commission_period'] == '2024-06']
    commission_summary = june_2024_data.groupby(['normalized_name'])['commission_amount'].sum().reset_index()
    
    # Find top 10 agents/agencies by commission amount
    top_10_agents_agencies = commission_summary.sort_values(by='commission_amount', ascending=False).head(10)

    print("Top 10 agents/agencies by commission payout for June 2024:")
    for index, row in top_10_agents_agencies.iterrows():
        print(f"{index + 1}. {row['normalized_name']}: ${row['commission_amount']:,.2f}")
